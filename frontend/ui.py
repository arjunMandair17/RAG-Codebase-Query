import streamlit as st
import requests

BASE_URL = "http://localhost:8000"

if "codebase_ingested" not in st.session_state:
    st.session_state.codebase_ingested = False

st.title("RAG-Based Github Repo Search")

if not st.session_state.codebase_ingested:
    github_url = st.text_input("Enter a GitHub URL:")
    st.caption("Large codebases may take several minutes to ingest while files are fetched and embedded.")

    if st.button("Search"):
        with st.spinner("Ingesting codebase... This may take a while for larger repos."):
            response = requests.post(f"{BASE_URL}/ingest", json={"github_url": github_url}, timeout=600)
        if response.status_code == 400:
            st.write("Invalid GitHub URL, please try again.")
        elif response.status_code == 500:
            st.write("Error ingesting codebase, please try again.")
        elif response.status_code == 200:
            st.write("Codebase ingested successfully")
            st.session_state.codebase_ingested = True
            st.rerun()
        else:
            st.write("Error ingesting codebase")
else:
    query = st.text_input("Enter a query:")
    if st.button("Delete context and ingest a new codebase"):
        with st.spinner("Clearing ingested codebase..."):
            response = requests.delete(f"{BASE_URL}/ingest", timeout=120)
        if response.status_code == 200:
            st.write("Codebase deleted successfully")
            st.session_state.codebase_ingested = False
            st.rerun()
        else:
            st.write("Error deleting codebase, please try again.")
    if st.button("Search"):
        with st.spinner("Searching codebase and generating an answer..."):
            response = requests.post(f"{BASE_URL}/retrieve", json={"query": query}, timeout=120)
        if response.status_code == 200:
            st.write(response.json()["answer"])
        else:
            st.write("Error retrieving answer, please try again.")