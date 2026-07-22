"""
Phase 5 — Streamlit frontend for the RAG Document Portal.
Talks to the FastAPI backend over HTTP; run the API first.

Run:
    uvicorn api.main:app --reload --port 8000      # in one terminal
    streamlit run frontend/app.py                  # in another
"""

import os
import time
import uuid

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
API_KEY = os.environ.get("PORTAL_API_KEY", "")
REQUEST_TIMEOUT = 120  # seconds — LLM calls can be slow, especially Ollama on CPU

# How many times to silently retry a request that fails with a transient,
# cold-start-shaped error (502/503, or a dropped connection) before actually
# showing the user an error. Render's free tier sleeps the service after 15
# minutes idle, and the first request after that can bounce a couple of
# times while the container wakes up — retrying here means a normal user
# just sees a slightly slower response instead of a scary error message.
COLD_START_RETRIES = 3
COLD_START_RETRY_DELAY_SECONDS = 8

st.set_page_config(page_title="RAG Document Portal", layout="wide")

if not API_KEY:
    st.sidebar.warning(
        "PORTAL_API_KEY is not set for this frontend — requests to the API will be rejected. "
        "Set it in your environment before running `streamlit run frontend/app.py`."
    )

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)
if "documents" not in st.session_state:
    st.session_state.documents = []
if "client_id" not in st.session_state:
    # One random ID per Streamlit session, used to scope which documents
    # this browser/device can see on the backend (see api/state.py's
    # client_documents). Without this, every device sharing the same
    # PORTAL_API_KEY would see every other device's uploaded documents —
    # your laptop and phone would show each other's files, which is exactly
    # what this was added to prevent.
    #
    # Known limitation: a hard browser refresh starts a new Streamlit
    # session, which generates a *new* client_id — so a refreshed tab will
    # show "no documents" even for files you uploaded moments ago in that
    # same browser. Re-uploading the same file is instant (the content is
    # deduplicated server-side), so this is a minor inconvenience rather
    # than a real loss of data, but worth knowing about.
    st.session_state.client_id = str(uuid.uuid4())

HEADERS = {}
if API_KEY:
    HEADERS["X-API-Key"] = API_KEY
HEADERS["X-Client-Id"] = st.session_state.client_id


def api_reachable_error(e: requests.RequestException) -> str:
    if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 401:
        return (
            "API rejected the request (401 Unauthorized) — PORTAL_API_KEY doesn't match what the "
            "API expects. Check that the value set for this Streamlit process matches your .env file exactly."
        )
    return f"Could not reach the API at {API_BASE} — is it running? ({e})"


def _is_cold_start_error(exc: Exception | None, resp: requests.Response | None) -> bool:
    """True for errors that look like Render's free-tier instance waking up
    from sleep, rather than a real, permanent failure."""
    if resp is not None and resp.status_code in (502, 503, 504):
        return True
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    return False


def request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Wraps requests.request with a few silent retries for cold-start-shaped
    failures (502/503/504, dropped connections). On the free tier, the first
    request after 15+ minutes of inactivity can bounce like this while the
    container wakes up — retrying here means the user just sees things take
    a little longer, rather than an alarming error on the very first click.
    Raises the underlying exception (or returns the last response as-is) if
    every attempt fails, so existing error handling downstream still works.
    """
    last_exc: requests.RequestException | None = None
    last_resp: requests.Response | None = None

    for attempt in range(COLD_START_RETRIES + 1):
        try:
            resp = requests.request(method, url, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            if attempt < COLD_START_RETRIES and _is_cold_start_error(e, None):
                time.sleep(COLD_START_RETRY_DELAY_SECONDS)
                continue
            raise

        if resp.status_code in (502, 503, 504) and attempt < COLD_START_RETRIES:
            last_resp = resp
            time.sleep(COLD_START_RETRY_DELAY_SECONDS)
            continue

        return resp

    # Exhausted retries — return the last response we got (if any) so
    # callers' existing status-code handling still applies.
    if last_resp is not None:
        return last_resp
    raise last_exc  # pragma: no cover — only reached if every attempt raised


def refresh_documents():
    try:
        resp = request_with_retry("GET", f"{API_BASE}/documents", headers=HEADERS, timeout=10)
        resp.raise_for_status()
        st.session_state.documents = resp.json()
    except requests.RequestException as e:
        st.sidebar.error(api_reachable_error(e))


def error_detail(resp: requests.Response) -> str:
    """Pull the FastAPI 'detail' field out of an error response, falling
    back to raw text if the body isn't the JSON shape we expect."""
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"


with st.sidebar:
    st.header("📁 Documents")

    uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
    if uploaded_file is not None:
        if st.button("Ingest document", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = request_with_retry(
                        "POST",
                        f"{API_BASE}/documents/upload",
                        files=files,
                        headers=HEADERS,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data["already_indexed"]:
                            st.info(f"'{uploaded_file.name}' was already indexed — reused the existing copy.")
                        else:
                            st.success(f"Indexed '{uploaded_file.name}' — {data['document']['num_chunks']} chunks.")
                        refresh_documents()
                    else:
                        st.error(f"Upload failed: {error_detail(resp)}")
                except requests.RequestException as e:
                    st.error(api_reachable_error(e))

    if st.button("Refresh list", use_container_width=True):
        refresh_documents()

    if not st.session_state.documents:
        refresh_documents()

    doc_labels = {
        d["source_id"]: f"{d['filename']} ({d['num_chunks']} chunks)" for d in st.session_state.documents
    }

    if not doc_labels:
        st.caption("No documents indexed yet — upload one above to get started.")

    selected_ids = st.multiselect(
        "Select document(s) to use",
        options=list(doc_labels.keys()),
        format_func=lambda sid: doc_labels.get(sid, sid),
    )

st.title("📄 RAG Document Portal")

tab_chat, tab_compare = st.tabs(["💬 Chat", "⚖️ Compare"])

with tab_chat:
    if not selected_ids:
        st.info("Select at least one document from the sidebar to start chatting.")
    else:
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

        question = st.chat_input("Ask a question about the selected document(s)...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = request_with_retry(
                            "POST",
                            f"{API_BASE}/chat",
                            json={
                                "question": question,
                                "document_ids": selected_ids,
                                "session_id": st.session_state.session_id,
                            },
                            headers=HEADERS,
                            timeout=REQUEST_TIMEOUT,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.session_id = data["session_id"]
                            answer = data["answer"]
                        else:
                            answer = f"⚠️ {error_detail(resp)}"
                    except requests.RequestException as e:
                        answer = f"⚠️ {api_reachable_error(e)}"

                    st.markdown(answer)
                    st.session_state.chat_history.append(("assistant", answer))

        if st.session_state.chat_history:
            if st.button("Clear conversation"):
                st.session_state.chat_history = []
                st.session_state.session_id = None
                st.rerun()

with tab_compare:
    if len(selected_ids) < 2:
        st.info("Select at least 2 documents from the sidebar to compare them.")
    else:
        topic = st.text_input("Topic to compare (e.g. 'revenue growth')")
        if st.button("Compare", disabled=not topic.strip()):
            with st.spinner("Comparing documents — this can take a moment..."):
                try:
                    resp = request_with_retry(
                        "POST",
                        f"{API_BASE}/compare",
                        json={"document_ids": selected_ids, "topic": topic},
                        headers=HEADERS,
                        timeout=REQUEST_TIMEOUT * 2,  # comparison touches multiple docs, give it more room
                    )
                    if resp.status_code == 200:
                        st.markdown(resp.json()["result"])
                    else:
                        st.error(f"Comparison failed: {error_detail(resp)}")
                except requests.RequestException as e:
                    st.error(api_reachable_error(e))
