import time

import requests
import streamlit as st

BASE_URL = "http://localhost:8000"

for key, default in {
    "codebase_ingested": False,
    "ingesting": False,
    "retrieving": False,
    "clearing": False,
    "pending_query": None,
    "last_answer": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def sync_ui_state() -> None:
    """Align session state with the API after a page reload."""
    if st.session_state.retrieving or st.session_state.clearing:
        return
    if st.session_state.codebase_ingested and not st.session_state.ingesting:
        return

    try:
        status = requests.get(f"{BASE_URL}/ingest/status", timeout=10).json()
        if status.get("state") == "running":
            st.session_state.ingesting = True
            st.session_state.codebase_ingested = False
            return
    except requests.RequestException:
        pass

    try:
        health = requests.get(f"{BASE_URL}/health/", timeout=10).json()
        if health.get("embedded_chunks", 0) > 0:
            st.session_state.codebase_ingested = True
            st.session_state.ingesting = False
    except requests.RequestException:
        pass


def poll_ingest_status() -> None:
    """Poll the API until background ingest completes or fails."""
    try:
        status = requests.get(f"{BASE_URL}/ingest/status", timeout=10).json()
    except requests.RequestException:
        st.warning("Lost connection to API while ingesting. Retrying...")
        time.sleep(2)
        st.rerun()
        return

    state = status.get("state")
    if state == "running":
        phase = status.get("phase", "working")
        embedded = status.get("chunks_embedded", 0)
        total = status.get("chunks_total") or "?"
        st.caption(f"{phase} — {embedded}/{total} chunks embedded")
        time.sleep(2)
        st.rerun()
    elif state == "complete":
        st.session_state.ingesting = False
        st.session_state.codebase_ingested = True
        st.success(f"Codebase ingested successfully ({status.get('chunks_stored', 0)} chunks)")
        st.rerun()
    elif state == "failed":
        st.session_state.ingesting = False
        st.error(status.get("error", "Ingest failed"))
    else:
        st.session_state.ingesting = False


def run_retrieve() -> None:
    """Call the retrieve API and store the answer in session state."""
    query = st.session_state.pending_query
    try:
        response = requests.post(
            f"{BASE_URL}/retrieve",
            json={"query": query},
            timeout=120,
        )
    except requests.RequestException:
        st.session_state.last_answer = None
        st.error("Could not reach API. Is the server running on port 8000?")
    else:
        if response.status_code == 200:
            st.session_state.last_answer = response.json()["answer"]
        else:
            st.session_state.last_answer = None
            st.error("Error retrieving answer, please try again.")

    st.session_state.retrieving = False
    st.session_state.pending_query = None


def run_clear() -> None:
    """Clear the vector DB and return to the ingest screen."""
    try:
        response = requests.delete(f"{BASE_URL}/ingest", timeout=120)
    except requests.RequestException:
        st.error("Could not reach API. Is the server running on port 8000?")
        st.session_state.clearing = False
        return

    if response.status_code == 200:
        st.session_state.codebase_ingested = False
        st.session_state.last_answer = None
        st.success("Codebase deleted successfully")
        st.rerun()
    elif response.status_code == 409:
        st.warning("Ingest is still running. Wait for it to finish before clearing.")
    else:
        st.error("Error deleting codebase, please try again.")

    st.session_state.clearing = False


st.title("RAG-Based Github Repo Search")

sync_ui_state()

if st.session_state.ingesting:
    with st.spinner("Ingesting codebase... This may take a while for larger repos."):
        poll_ingest_status()

elif not st.session_state.codebase_ingested:
    github_url = st.text_input("Enter a GitHub URL:", disabled=st.session_state.ingesting)
    st.caption("Large codebases may take 10+ minutes to ingest while files are fetched and embedded.")

    if st.button("Search", disabled=st.session_state.ingesting):
        if not github_url:
            st.write("Please enter a GitHub URL.")
        else:
            try:
                response = requests.post(
                    f"{BASE_URL}/ingest",
                    json={"github_url": github_url},
                    timeout=30,
                )
            except requests.RequestException:
                st.write("Could not reach API. Is the server running on port 8000?")
            else:
                if response.status_code == 400:
                    st.write("Invalid GitHub URL, please try again.")
                elif response.status_code in (409, 202):
                    st.session_state.ingesting = True
                    st.rerun()
                else:
                    st.write("Error starting ingest, please try again.")

else:
    busy = st.session_state.retrieving or st.session_state.clearing

    query = st.text_input("Enter a query:", disabled=busy)

    if st.button("Delete context and ingest a new codebase", disabled=busy):
        st.session_state.clearing = True
        st.rerun()

    if st.button("Search", disabled=busy):
        if not query:
            st.write("Please enter a query.")
        else:
            st.session_state.pending_query = query
            st.session_state.retrieving = True
            st.rerun()

    if st.session_state.clearing:
        with st.spinner("Clearing ingested codebase..."):
            run_clear()

    if st.session_state.retrieving and st.session_state.pending_query:
        with st.spinner("Searching codebase and generating an answer..."):
            run_retrieve()

    if st.session_state.last_answer and not st.session_state.retrieving:
        st.write(st.session_state.last_answer)
