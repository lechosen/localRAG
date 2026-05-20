import gradio as gr
from src.ingest import ingest
from src.rag_chain import ask
import os

def handle_upload(file):
    if file is None:
        return "Please upload a file"
    try:
        original_name = os.path.basename(file.name)
        n = ingest(file.name, display_name=original_name)
        return f"Ingested {n} chunks from: {original_name}"
    except Exception as e:
        return f"Error: {e}"

def handle_url(url: str):
    if not url.strip():
        return "Please enter a URL"
    try:
        n = ingest(url.strip(), display_name=url.strip())
        return f"Ingested {n} chunks from: {url}"
    except Exception as e:
        return f"Error: {e}"

def handle_question(question: str):
    if not question.strip():
        return "", ""
    result = ask(question)
    sources_text = "\n".join(f"- {s}" for s in result["sources"])
    return result["answer"], sources_text

with gr.Blocks(title="Local RAG Q&A") as demo:
    gr.Markdown("# Local RAG Q&A System")
    with gr.Tab("Document Ingestion"):
        file_input = gr.File(label="Upload PDF / TXT", file_types=[".pdf", ".txt", ".md"])
        file_status = gr.Textbox(label="Status")
        # Use .upload event (fires after upload completes) instead of a separate button
        # to avoid a race condition where the button click could use a stale file reference
        file_input.upload(handle_upload, inputs=file_input, outputs=file_status)
        url_input = gr.Textbox(label="or enter webpage URL")
        url_btn = gr.Button("Ingest")
        url_status = gr.Textbox(label="Status")
        url_btn.click(handle_url, inputs=url_input, outputs=url_status)
    with gr.Tab("Q&A"):
        question_input = gr.Textbox(label="Question", lines=2)
        ask_btn = gr.Button("Ask", variant="primary")
        answer_output = gr.Textbox(label="Answer", lines=6)
        sources_output = gr.Textbox(label="Sources", lines=3)
        ask_btn.click(handle_question, inputs=question_input,
                      outputs=[answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()