# Local RAG Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5%2B-FF6F61)](https://www.trychroma.com/)

A local-first Retrieval-Augmented Generation (RAG) application for asking grounded questions about PDF documents. PDFs, embeddings, and the vector index remain on the local machine; Gemini is used for embeddings and answer generation.

## Architecture

1. `app.py` stores uploaded PDFs in `uploads/` and provides the Streamlit interface.
2. `ingestion.py` loads PDFs with `PyPDFLoader`, splits text into 1,000-character chunks with 150-character overlap, and adds page/file metadata.
3. `GoogleGenerativeAIEmbeddings` uses `gemini-embedding-001` (the Google API resource is commonly displayed as `models/gemini-embedding-001`).
4. ChromaDB persists the embeddings in the `rag_knowledge_base` collection under `chroma_db/`.
5. `rag_engine.py` retrieves up to 4 chunks with relevance scores. Results below the default relevance threshold of `0.55` are rejected before the chat model is called.
6. `ChatGoogleGenerativeAI` is configured with `gemini-3.6-flash` and `temperature=0.2` to answer only from retrieved context and include file/page citations. Google applies fixed sampling defaults for this model, so the custom temperature argument may be ignored at runtime.

## Features

- Multi-PDF upload and automatic indexing from the Document Vault.
- Duplicate prevention for documents already in the Chroma collection.
- Search across all documents or filter to one PDF.
- Source citations with file name, page number, snippet, and relevance score.
- Strict grounding and a no-relevant-context guardrail that avoids unnecessary LLM calls.
- Local collection reset control that clears the Chroma index, uploaded PDFs, and chat history.
- Script-relative paths, so the application can be launched from any working directory.

## Project Structure

```text
pdfreader/
├── app.py              # Streamlit UI and document management (repository source)
├── ingestion.py        # PDF parsing, chunking, embeddings, and indexing (repository source)
├── rag_engine.py       # Retrieval, guardrails, citations, and Gemini answers (repository source)
├── requirements.txt    # Python dependencies (repository source)
├── README.md           # Project documentation (repository source)
├── .gitignore          # Ignore rules (repository source)
├── .env                # GEMINI_API_KEY; local secret, ignored and never committed
├── uploads/            # Uploaded PDFs; local runtime data, ignored
├── chroma_db/          # Chroma SQLite/index files; local runtime data, ignored
├── __pycache__/        # Python bytecode cache; ignored
└── .streamlit/         # Optional local Streamlit config; ignored
```

The source files and documentation should be committed. The `.env`, `uploads/`, `chroma_db/`, `__pycache__/`, and `.streamlit/` entries are excluded by the current `.gitignore` and should remain local.

## Prerequisites

- Python 3.10 or newer.
- A Google AI Studio API key with access to the configured Gemini models.
- Network access while embedding documents or generating answers. PDF files and Chroma storage remain local.

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked by a Windows execution-policy restriction, run this in the same PowerShell session before activating:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

Create a `.env` file in the project root and add the required key:

```dotenv
GEMINI_API_KEY=your-gemini-api-key
```

Do not commit `.env` or expose the key in source control.

## Run

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Usage

1. Upload one or more PDFs in the **Document Vault** sidebar.
2. Wait for **Processing & embedding document...** to finish.
3. Choose **All Documents** or a specific PDF under **Search scope**.
4. Ask a question in the chat input.
5. Expand **View Sources & Citations** to inspect the supporting pages and relevance scores.
6. Use **Clear All Documents & Reset Index** to remove local uploads, reset the Chroma collection, and clear the conversation.

## Retrieval and Guardrails

Chroma returns normalized relevance scores from `similarity_search_with_relevance_scores`, where higher is better and scores are expected to be between `0.0` and `1.0`. The default `0.55` cutoff prevents unrelated queries from reaching Gemini. The retrieval defaults are:

| Setting | Default |
| --- | ---: |
| Chunk size | 1,000 characters |
| Chunk overlap | 150 characters |
| Retrieved chunks (`top_k`) | 4 |
| Minimum relevance score | 0.55 |
| Chat temperature | 0.2 (configured; Google may ignore custom temperature for `gemini-3.6-flash` because the model uses fixed sampling defaults) |

## Dependencies

The pinned minimum package requirements are listed in `requirements.txt`:

- `langchain`, `langchain-community`, and `langchain-text-splitters`
- `langchain-google-genai`
- `langchain-chroma` and `chromadb`
- `pypdf`
- `python-dotenv`
- `streamlit`

## Cost and Quota

The application has no hosting, object-storage, or managed-vector-database charges: PDFs and Chroma data are stored locally, and Streamlit runs locally. Gemini calls use Google AI Studio's available free-tier quotas. Free access is subject to Google's current rate limits, daily quotas, model availability, and terms; usage beyond those limits may require billing. API calls are still made for embeddings and accepted answer-generation requests.

## Troubleshooting / FAQ

### `GEMINI_API_KEY` is missing

Create a project-root `.env` file with `GEMINI_API_KEY=...`, restart Streamlit, and verify that the key is valid for Google AI Studio.

### A Gemini model is unavailable or returns a 404

Confirm that the configured model is enabled for the API key and available in the account's region/project. The current code uses `gemini-embedding-001` and `gemini-3.6-flash`; model availability and names can change, so update both the code and this documentation together when Google publishes a migration.

### A PDF fails with an invalid header or no extractable text

Ensure the upload is a real, readable PDF whose first four bytes are `%PDF`. Re-export password-protected, HTML-renamed, truncated, or corrupted files as ordinary PDFs before uploading.

### The app shows no relevant information

The query may be below the `0.55` relevance cutoff. Try wording the question closer to the document text, select the correct PDF in **Search scope**, or verify that the document contains extractable text.

### The local index is stale or corrupted

Use **Clear All Documents & Reset Index** in the sidebar, then upload the PDFs again. This deletes the Chroma collection and local uploads.

### Chroma or dependency import errors occur

Activate the project virtual environment and reinstall with `python -m pip install -r requirements.txt`. Use Python 3.10 or newer and restart the Streamlit process after dependency changes.

## License

No `LICENSE` file is currently included in this repository. Add an explicit license before distributing the project or accepting external contributions.