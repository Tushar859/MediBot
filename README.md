# MediBot – AI-Powered Medical PDF RAG Chatbot

MediBot is an AI-powered medical PDF chatbot that allows users to upload medical reference documents and ask questions in natural language. It uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded PDFs and generate context-aware answers using a Large Language Model (LLM).

Users do not need their own AI account or API token. The application uses a centrally configured Hugging Face API token.

## Key Features

- **PDF Upload:** Upload medical reference PDFs and interact with their content.
- **RAG-Based Question Answering:** Retrieves relevant passages before generating responses.
- **Semantic Search:** Uses Sentence Transformers embeddings and FAISS for efficient similarity search.
- **LLM Integration:** Uses Qwen2.5-7B-Instruct through Hugging Face Inference Providers.
- **Source References:** Displays source page numbers to help users verify retrieved information.
- **OCR Support:** Extracts text from scanned PDF pages using Tesseract OCR.
- **Flexible PDF Processing:** Supports quick-start page ranges, custom page ranges, and full-document processing for PDFs exceeding 50 pages.
- **Interactive Interface:** Built with Streamlit, including welcome cards and sample questions.
- **Session Management:** Maintains uploaded PDF and conversation history within the Streamlit session.
- **Question Limit:** Limits each session to 10 questions as a demo safeguard.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Frontend and application interface |
| LangChain | PDF loading and document processing |
| PyPDFLoader | PDF text extraction |
| Tesseract OCR | Text extraction from scanned pages |
| Sentence Transformers | Text embeddings |
| FAISS | Vector storage and similarity search |
| Hugging Face | LLM inference |
| Qwen2.5-7B-Instruct | Answer generation |

## Project Structure

```text
MediBot/
│
├── app.py              # Main Streamlit application
├── ui.py               # User interface and styling
├── rag.py              # RAG pipeline and retrieval logic
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for deployment
└── README.md           # Project documentation
```

## How It Works

MediBot follows a Retrieval-Augmented Generation pipeline:

1. **PDF Upload:** User uploads a medical reference PDF.
2. **Text Extraction:** Extracts text using PDF processing and OCR for scanned pages.
3. **Text Chunking:** Splits extracted content into smaller chunks of 500 characters with a 50-character overlap.
4. **Embedding Generation:** Converts text chunks into vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
5. **Vector Indexing:** Stores embeddings in a FAISS vector index.
6. **Similarity Search:** Retrieves the top 3 most relevant passages based on the user's question.
7. **Answer Generation:** Sends the retrieved context to Qwen2.5-7B-Instruct through Hugging Face Inference Providers.
8. **Response Display:** Displays the generated answer along with source page references.

### Architecture

```text
        User
         |
         v
    Upload Medical PDF
         |
         v
    Text Extraction
    (PyPDFLoader / OCR)
         |
         v
    Text Chunking
   (500 / 50 overlap)
         |
         v
  Sentence Transformers
    (Embeddings)
         |
         v
      FAISS Index
         |
         v
    User Question
         |
         v
   Similarity Search
    (Top 3 Chunks)
         |
         v
  Qwen2.5-7B-Instruct
         |
         v
   Generated Answer
         |
         v
  Answer + Source Pages
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tushar859/MediBot.git
cd MediBot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Hugging Face API Token

Create a Hugging Face access token with **Inference Providers** permission.

For Windows PowerShell:

```powershell
$env:HF_TOKEN="your_huggingface_token"
```

For deployment, configure `HF_TOKEN` in your hosting provider's private secrets or environment variables.

**Important:** Never upload your API token or `.env` file to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Deployment

MediBot can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Connect your GitHub repository to Streamlit Community Cloud.
3. Configure `HF_TOKEN` in the application's Secrets settings.
4. Deploy the application.

The `packages.txt` file installs Tesseract OCR on Streamlit Community Cloud. For local Windows OCR support, install Tesseract separately and ensure it is accessible through your system PATH.

## Configuration and Model Details

| Component | Configuration |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Database | FAISS |
| Retrieval Strategy | Similarity Search |
| Retrieved Chunks | Top 3 |
| Chunk Size | 500 characters |
| Chunk Overlap | 50 characters |
| LLM | `Qwen/Qwen2.5-7B-Instruct` |
| LLM Provider | Hugging Face Inference Providers |
| Interface | Streamlit |

**Note:** Model provider availability, account credits, and billing requirements may affect inference.

## Future Improvements

- Conversation memory across multiple sessions.
- Support for multiple PDF uploads.
- Improved medical document retrieval and ranking.
- Downloadable chat summaries.
- More advanced source citations and document highlighting.
- Support for additional open-source LLMs.

## Disclaimer

MediBot is an educational and informational project developed to demonstrate Retrieval-Augmented Generation and AI-powered document question answering.

It is not a medical diagnostic tool and does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## Author

**Tushar Mishra**

B.Tech – Computer Science and Engineering (AI & ML)

GitHub: [@Tushar859](https://github.com/Tushar859)

---

⭐ If you find this project interesting, consider giving the repository a star!
