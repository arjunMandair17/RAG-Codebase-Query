import asyncio
import os
import re
from urllib.parse import urlparse
import json

import httpx
from tree_sitter_language_pack import get_language, get_parser

GITHUB_API = "https://api.github.com"
MAX_CONCURRENT_FETCHES = 20
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
    ".json": "json"
}

SKIP_INDEX_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "package.json",
    "readme.md",
}


def _should_skip_file(path: str) -> bool:
    """Skip lockfiles, package manifests, and README — they drown out source code in retrieval."""
    return os.path.basename(path).lower() in SKIP_INDEX_FILES

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


async def _fetch_file_tree(
    client: httpx.AsyncClient, owner: str, repo: str, branch: str, headers: dict[str, str]
) -> list[str]:
    """List all relevant file paths in a repo via the Git Trees API."""
    resp = await client.get(
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


async def _fetch_file_content(
    client: httpx.AsyncClient, owner: str, repo: str, branch: str, path: str
) -> str:
    """Download a single file's text content via raw.githubusercontent.com."""
    resp = await client.get(
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
    )
    resp.raise_for_status()
    return resp.text


async def parse_code(github_url: str) -> list[dict]:
    """
    Fetch text files from a public GitHub repo in parallel.

    Returns a list of dicts with keys: path, content, extension, language.
    """
    owner, repo, branch = parse_github_url(github_url)
    headers = _github_headers(None)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=MAX_CONCURRENT_FETCHES),
    ) as client:
        if not branch:
            resp = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=headers)
            resp.raise_for_status()
            branch = resp.json()["default_branch"]

        paths = await _fetch_file_tree(client, owner, repo, branch, headers)

        async def fetch_file(path: str) -> dict:
            async with semaphore:
                content = await _fetch_file_content(client, owner, repo, branch, path)
            ext = os.path.splitext(path)[1].lower()
            return {
                "path": path,
                "content": content,
                "extension": ext,
                "language": LANGUAGES.get(ext),
            }

        return list(
            await asyncio.gather(
                *[
                    fetch_file(path)
                    for path in paths
                    if not _should_skip_file(path)
                ]
            )
        )



CHUNK_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
    "java": {"method_declaration", "class_declaration"},
    "rust": {"function_item", "impl_item", "struct_item"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
}


def _json_chunks(value, path: str, language: str, key: str | None = None, second_key: str | None = None) -> list[dict]:
    """Flatten JSON values into chunk records."""
    chunks: list[dict] = []

    if isinstance(value, list):
        for index, item in enumerate(value):
            chunks.extend(_json_chunks(item, path, language, key=key, second_key=str(index)))
        return chunks

    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            chunks.extend(_json_chunks(nested_value, path, language, key=key or nested_key, second_key=nested_key))
        return chunks

    chunk = {
        "text": json.dumps(value, indent=2),
        "path": path,
        "language": language,
        "type": "json",
    }
    if key is not None:
        chunk["key"] = key
    if second_key is not None:
        chunk["second_key"] = second_key
    chunks.append(chunk)
    return chunks

def chunk_code(code: str, language: str | None = None, path: str = "", extension: str = "") -> list[dict]:
    """Chunk code into text + metadata dicts ready for embedding and vector DB storage."""

    if not language:
        language = LANGUAGES.get(extension) or "text"
    
    raw = code.encode()
    chunks, work = [], [get_parser(language).parse(code).root_node()] if language in CHUNK_TYPES else []

    ## special case for json files: split into key-value pairs
    if extension == ".json":
        return _json_chunks(json.loads(code), path, language)

    ## special case for markdown files: split into chunks based on headers
    if extension == ".md":
        curChunk = ""
        for line in code.split("\n"):
            if line.isspace():
                continue    ## skip empty lines
            if line.startswith("#"):
                if curChunk.strip():
                    chunks.append({"text": curChunk, "path": path, "language": language, "type": "text"})
                curChunk = line + "\n"
            else:
                curChunk += line + "\n"
        if curChunk.strip():
            chunks.append({"text": curChunk, "path": path, "language": language, "type": "text"})
        return chunks

    ## traverse the tree through a stack and append valid chunks to the list
    while work:
        node = work.pop()
        if node.kind() in CHUNK_TYPES[language]:
            chunks.append({"text": raw[node.start_byte():node.end_byte()].decode(), "path": path, "language": language, "type": node.kind()})
        else:
            work.extend(node.child(i) for i in range(node.child_count()))        

    ## fallback: if there were no valid chunks, split code on double newlines and append as text chunks
    if not chunks:
        for part in re.split(r"\n{2,}", code.strip()):
            if part.strip():
                chunks.append({"text": part.strip(), "path": path, "language": language, "type": "text"})
    return chunks
