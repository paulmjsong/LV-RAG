# LV-RAG

LV-RAG is a general-purpose RAG model designed for dynamic and flexible use. It allows users to:

- **Select their preferred LLM** in real time.
- **Upload one or more PDFs** for document-based question answering.
- **Ask queries** and receive responses based on the uploaded PDFs.
- **Fallback to the LLM's pretrained knowledge** if no PDFs are provided.

Built with **Gradio**, LV-RAG provides an intuitive web-based interface for seamless interaction.

## Features
✅ **Dynamic LLM Selection** – Choose an LLM at runtime.  
✅ **PDF Processing** – Extract and retrieve relevant content.  
✅ **Hybrid Knowledge Source** – Combines document-based retrieval with pretrained knowledge.  
✅ **User-Friendly Interface** – Simple web-based UI powered by Gradio.  

## Installation & Usage

### Prerequisites
- Python 3.8+
- Required dependencies (install via `requirements.txt`)

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-repo/lv-rag.git
cd lv-rag
pip install -r requirements.txt
```

### Running LV-RAG
Start the Gradio app:
```bash
python app.py
```
This will launch a Gradio web interface, accessible via your browser.
