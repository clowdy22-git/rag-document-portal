"""
Phase 5 — Streamlit frontend for the RAG Document Portal.
Talks to the FastAPI backend over HTTP; run the API first.

Run:
    uvicorn api.main:app --reload --port 8000      # in one terminal
    streamlit run frontend/app.py                  # in another
"""

import os

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
API_KEY = os.environ.get("PORTAL_API_KEY", "")
REQUEST_TIMEOUT = 120  # seconds — LLM calls can be slow, especially Ollama on CPU

AUTH_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

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


def api_reachable_error(e: requests.RequestException) -> str:
    return f"Could not reach the API at {API_BASE} — is it running? ({e})"


def refresh_documents():
    try:
        resp = requests.get(f"{API_BASE}/documents", headers=AUTH_HEADERS, timeout=10)
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
                    resp = requests.post(
                        f"{API_BASE}/documents/upload", files=files, headers=AUTH_HEADERS, timeout=REQUEST_TIMEOUT
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
                        resp = requests.post(
                            f"{API_BASE}/chat",
                            json={
                                "question": question,
                                "document_ids": selected_ids,
                                "session_id": st.session_state.session_id,
                            },
                            headers=AUTH_HEADERS,
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
                    resp = requests.post(
                        f"{API_BASE}/compare",
                        json={"document_ids": selected_ids, "topic": topic},
                        headers=AUTH_HEADERS,
                        timeout=REQUEST_TIMEOUT * 2,  # comparison touches multiple docs, give it more room
                    )
                    if resp.status_code == 200:
                        st.markdown(resp.json()["result"])
                    else:
                        st.error(f"Comparison failed: {error_detail(resp)}")
                except requests.RequestException as e:
                    st.error(api_reachable_error(e))
