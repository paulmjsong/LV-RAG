import gradio as gr
import os, json, datetime, time, faiss, fitz, openai
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


HF_API_KEY = "YOUR_HF_API_KEY"  # Set your Hugging Face API key here
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Set your OpenAI API key here

MODELS = {
    "DeepSeek R1": {"model_name": "deepseek-ai/DeepSeek-R1", "provider": "novita"},
    "Gemma 3": {"model_name": "google/gemma-3-27b-it", "provider": "hf-inference"},
    "Llama 3.3": {"model_name": "meta-llama/Llama-3.3-70B-Instruct", "provider": "hf-inference"},
    "o1-mini": {"model_name": "o1-mini", "provider": "openai"},
}

EMBED_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DIMENSION = 384
INDEX = faiss.IndexFlatL2(DIMENSION)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
vectorstore = None

chat_history = []


# --------- (A) 모델 및 InferenceClient 설정 ---------
model = MODELS["DeepSeek R1"]["model_name"]

client = InferenceClient(
    provider=MODELS["DeepSeek R1"]["provider"],
    api_key=HF_API_KEY,
)


# --------- (B) PDF 텍스트 추출 ---------
def before_processing():
    return gr.update(visible=False), gr.update(visible=True)

def extract_text_from_pdf(pdf):
    doc = fitz.open(pdf)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.strip()

def add_pdfs_to_db(uploaded_pdfs, state):
    global vectorstore
    if uploaded_pdfs is None:
        vectorstore = None
        return state, gr.update(visible=True), gr.update(visible=False)
    
    start_time = time.time()
    
    print(f"PDFs received: {len(uploaded_pdfs)}")
    for pdf in uploaded_pdfs:
        print(f"  {pdf}")
    
    # Determine PDFs to add or remove
    pdfs_to_remove = [pdf for pdf in state if pdf not in uploaded_pdfs]
    pdfs_to_add = [pdf for pdf in uploaded_pdfs if pdf not in state]
    num_to_remove = len(pdfs_to_remove)
    num_to_add = len(pdfs_to_add)
    num_to_change = num_to_remove + num_to_add

    # Remove old PDFs from DB
    if num_to_remove != 0:
        for i, pdf in enumerate(pdfs_to_remove):
            print(f"Removing {i+1}/{num_to_remove} PDFs...")
            state.pop(pdf)
    
    # Add new PDFs to DB
    if num_to_add == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processed {num_to_change} PDFs in {elapsed_time:.2f} seconds")
        return state, gr.update(visible=True), gr.update(visible=False)
    new_docs = []
    for i, pdf in enumerate(pdfs_to_add):
        # Load content of new PDFs
        print(f"Adding {i+1}/{num_to_add} PDFs...")
        print("  Extracting text...")
        text = extract_text_from_pdf(pdf)
        doc = Document(page_content=text, metadata={"source": pdf})

        # Split content into smaller docs
        print("  Splitting into chunks...")
        docs = text_splitter.split_documents([doc])
        new_docs.extend(docs)

    # Save docs into vectorstore
    print("Saving to vectorstore...")
    if vectorstore is None:
        vectorstore = FAISS.from_documents(new_docs, EMBED_MODEL)
    else:
        vectorstore.add_documents(new_docs)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {num_to_change} PDFs in {elapsed_time:.2f} seconds")
    state.extend(pdfs_to_add)
    return state, gr.update(visible=True), gr.update(visible=False)


# --------- (C) 메인 채팅 함수 ---------
def chat(query, history, top_k=3):
    
    global vectorstore, retriever
    start_time = time.time()

    if history is None:
        history = []

    # (1) 여러 PDF 파일에서 관련 텍스트 추출
    context = ""
    if vectorstore is not None:
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

    # (2) 사용자 메시지 구성
    combined_message = {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    history.append(combined_message)

    user_message = {"role": "user", "content": query}
    chat_history.append(user_message)

    messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
   
    # (3) 모델에 대화 내역 전달
    print("Inquiring LLM...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    
    # (4) 모델 답변 전처리
    def process_think_tags(text):
        print("Processing response...")
        if "<think>" in text or "</think>" in text:
            return text.replace("<think>", "&lt;think&gt;").replace("</think>", "&lt;/think&gt;")
        return text
    
    bot_message = response.choices[0].message
    bot_message.content = process_think_tags(bot_message.content)
    bot_message = {"role": bot_message.role, "content": bot_message.content}
    history.append(bot_message)
    chat_history.append(bot_message)
   
    # (5) JSON 파일로 대화 기록 저장
    save_history(history)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Responded to user query in {elapsed_time:.2f} seconds")
    return history, chat_history


# --------- (D) 유틸 함수들 ---------
def save_history(history):
    """대화 기록(history)을 JSON 파일로 저장"""
    folder = "./chat_logs"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"chat_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def extract_text(content):
    """content가 문자열일 때 그대로 반환. list나 다른 형태면 문자열로 변환"""
    if isinstance(content, list):
        return " ".join(str(item) for item in content).strip()
    elif isinstance(content, str):
        return content.strip()
    else:
        return str(content)

def change_model(model_name):
    """사용자 선택에 따라 모델 변경"""
    global model, client
    model = MODELS[model_name]["model_name"]
    provider = MODELS[model_name]["provider"]

    if provider == "openai":
        client = openai.Client(
            api_key=OPENAI_API_KEY
        )
    else:
        client = InferenceClient(
            provider=provider,
            api_key=HF_API_KEY,
        )
    print(f"Now using:\n  Model: {model}\n  Provider: {provider}")

def reset():
    """대화 및 PDF 업로드 내역 삭제"""
    global chat_history, vectorstore
    chat_history = []
    return [], [], []


# --------- (E) Gradio UI 구성 ---------
css = """
.row-custom {
    height: 90vh !important;
    padding-top: 16px;
}
.col-custom {
    height: 100% !important;
}
.box-custom {
    flex-grow: 1;
    overflow: auto !important;
}
.selected {
    background-color: #43B02A !important;
    color: white !important;
}
footer {
    visibility: hidden !important;
}
"""

js = """
window.highlightButton = function(selected_id) {
    let selectedButton = document.getElementById(selected_id);
    if (selectedButton) {
        let buttons = document.querySelectorAll('button');
        buttons.forEach(btn => btn.classList.remove('selected'));
        selectedButton.classList.add('selected');
    }
}
"""

with gr.Blocks(title="LiberVance RAG", css=css, js=js, fill_height=True) as demo:
    
    upload_state = gr.State([])    # 업로드한 PDF 기록 (업로드 리스트)
    history = gr.State([])   # 대화 전체 기록 (메시지 리스트)

    with gr.Row(elem_classes=["row-custom"]):
        # Input column
        with gr.Column(elem_classes=["col-custom"]):
            gr.Markdown("## LiberVance RAG")
            gr.Markdown(
                "여러 개의 PDF 파일을 업로드하면, 각 PDF에서 텍스트를 추출하고 모델이 참고할 수 있도록 전송합니다. "
                "단, PDF가 너무 크거나 여러 개면 토큰 초과가 발생할 수 있으므로 필요시 요약 또는 문서 검색 전략을 도입하세요."
            )
            pdf_uploader = gr.Files(
                label="Upload PDFs (optional, multiple allowed)",
                file_types=[".pdf"],
                elem_classes=["box-custom"],
            )
            upload_status = gr.Textbox(
                value="Processing PDFs...",
                visible=False,
                interactive=False,
                show_label=False,
                elem_classes=["box-custom"],
            )
            txt = gr.Textbox(
                label="Your Prompt",
                placeholder="Enter your question or prompt",
            )
            with gr.Row():
                send_btn = gr.Button("Send")
                reset_btn = gr.Button("Erase history")
        # Output column
        with gr.Column(elem_classes=["col-custom"]):
            with gr.Row():
                model_btns = []
                for i, (model_name, details) in enumerate(MODELS.items()):
                    btn_classes = "selected" if i == 0 else ""
                    model_btn = gr.Button(model_name, elem_id=f"btn{i+1}", elem_classes=btn_classes)
                    model_btn.click(lambda model=model_name: change_model(model), js=f"() => highlightButton('btn{i+1}')")
                    model_btns.append(model_btn)
            chatbot = gr.Chatbot(
                type="messages",
                elem_classes=["box-custom"],
            )
        # Event listeners
        pdf_uploader.change(fn=before_processing, inputs=[], outputs=[pdf_uploader, upload_status], queue=False)
        pdf_uploader.change(fn=add_pdfs_to_db, inputs=[pdf_uploader, upload_state], outputs=[upload_state, pdf_uploader, upload_status])
        send_btn.click(chat, inputs=[txt, history], outputs=[history, chatbot])
        reset_btn.click(reset, outputs=[upload_state, history, chatbot])

demo.launch()
