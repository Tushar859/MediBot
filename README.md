# MediBot

A simple medical PDF chatbot. Visitors upload a reference PDF and ask questions; they do not need their own AI account or API token.

## Project files

- `app.py` controls the page flow and remembers the current PDF and chat in Streamlit session state.
- `ui.py` contains the page styling, welcome cards, medical note, and source display.
- `rag.py` reads PDFs, creates the FAISS search index, retrieves passages, and asks the language model for an answer.
- `requirements.txt` lists the Python packages. `packages.txt` installs Tesseract OCR on Streamlit Community Cloud.

The app flow is: upload a PDF → `rag.py` extracts and indexes it → `app.py` gets the question → `rag.py` retrieves passages and generates an answer → `ui.py` displays the answer and source pages.

## Run locally

Install the packages and start the app:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Configure the app once

The app owner provides one Hugging Face token with **Inference Providers** permission. Keep it private.

- **For a deployed app:** add `HF_TOKEN` in the hosting provider's private secrets/settings.
- **For local use:** set the `HF_TOKEN` environment variable before starting Streamlit.

Never commit or publish the token. Once configured, visitors only upload their PDF and chat. The app uses Sentence Transformers embeddings, FAISS top-three retrieval, and the open `Qwen/Qwen2.5-7B-Instruct` model through Hugging Face Inference Providers. Provider availability and account credits or billing may apply.

MediBot displays source page numbers, offers sample questions, and limits each session to 10 questions. PDFs over 50 pages offer a quick-start range, a custom page range, or full-document processing so visitors can try a relevant section without waiting for the entire file. Scanned PDF pages use Tesseract OCR; `packages.txt` installs it on Streamlit Community Cloud. On Windows, install Tesseract separately to use OCR locally. The per-session question cap is a simple demo safeguard, not a global quota.

MediBot is an educational demo, not a substitute for professional medical advice.
