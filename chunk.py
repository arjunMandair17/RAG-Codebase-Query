import os
import re
from urllib.parse import urlparse

import httpx
from tree_sitter_language_pack import get_language, get_parser

GITHUB_API = "https://api.github.com"

# Extensions worth indexing for RAG over a codebase
RELEVANT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".rs", ".go", ".md", ".json",
    ".yaml", ".yml", ".toml", ".txt",
}

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".rs": "rust",
    ".go": "go",
}

# Skip common non-source directories
SKIP_DIRS = re.compile(
    r"(^|/)(\.git|node_modules|__pycache__|\.venv|venv|dist|build|vendor|target)(/|$)"
)


def parse_github_url(url: str) -> tuple[str, str, str | None]:
    """Extract owner, repo, and optional branch from a GitHub repo URL."""
    parts = [p for p in urlparse(url.rstrip("/")).path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")

    owner, repo = parts[0], parts[1].removesuffix(".git")
    branch = parts[3] if len(parts) > 3 and parts[2] == "tree" else None
    return owner, repo, branch


def _github_headers(token: str | None) -> dict[str, str]:
    """Build request headers, using GITHUB_TOKEN when available for higher rate limits."""
    headers = {"Accept": "application/vnd.github+json"}
    token = token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_file_tree(
    client: httpx.Client, owner: str, repo: str, branch: str, headers: dict[str, str]
) -> list[str]:
    """List all relevant file paths in a repo via the Git Trees API."""
    resp = client.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
        headers=headers,
    )
    resp.raise_for_status()

    paths = []
    for item in resp.json().get("tree", []):
        if item.get("type") != "blob":
            continue
        path = item["path"]
        if SKIP_DIRS.search(path):
            continue
        if os.path.splitext(path)[1].lower() not in RELEVANT_EXTENSIONS:
            continue
        paths.append(path)
    return paths


def _fetch_file_content(
    client: httpx.Client, owner: str, repo: str, branch: str, path: str
) -> str:
    """Download a single file's text content via raw.githubusercontent.com."""
    resp = client.get(
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
    )
    resp.raise_for_status()
    return resp.text


def parse_code(github_url: str) -> list[dict]:
    """
    Fetch text files from a public GitHub repo.

    Returns a list of dicts with keys: path, content, extension, language.
    language is set for code files (for tree-sitter later); None for plain text/json/md.
    """
    owner, repo, branch = parse_github_url(github_url)
    headers = _github_headers(None)

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        if not branch:
            resp = client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=headers)
            resp.raise_for_status()
            branch = resp.json()["default_branch"]

        paths = _fetch_file_tree(client, owner, repo, branch, headers)

        files = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            files.append({
                "path": path,
                "content": _fetch_file_content(client, owner, repo, branch, path),
                "extension": ext,
                "language": LANGUAGES.get(ext),
            })
        return files



CHUNK_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
    "java": {"method_declaration", "class_declaration"},
    "rust": {"function_item", "impl_item", "struct_item"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
}

def chunk_code(code: str, language: str | None = None, path: str = "") -> list[dict]:
    """Chunk code into text + metadata dicts ready for embedding and vector DB storage."""
    raw = code.encode()
    chunks, work = [], [get_parser(language).parse(code).root_node()] if language in CHUNK_TYPES else []

    while work:
        node = work.pop()
        if node.kind() in CHUNK_TYPES[language]:
            chunks.append({"text": raw[node.start_byte():node.end_byte()].decode(), "path": path, "language": language, "type": node.kind()})
        else:
            work.extend(node.child(i) for i in range(node.child_count()))

    if not chunks:
        for part in re.split(r"\n{2,}", code.strip()):
            if part.strip():
                chunks.append({"text": part.strip(), "path": path, "language": language, "type": "text"})
    return chunks
