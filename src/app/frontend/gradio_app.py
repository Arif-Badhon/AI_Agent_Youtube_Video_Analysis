import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.backend.agent import generate_summary, answer_question
from app.backend.utils import get_transcript

css = """
.gradio-container {max-width: 900px!important}
footer {visibility: hidden}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as app:
    gr.Markdown("# YouTube AI Analyzer 🎥")
    
    with gr.Tab("Video Analysis"):
        with gr.Row():
            url_input = gr.Textbox(label="YouTube URL", placeholder="Paste video URL here...")
            summary_output = gr.Textbox(label="Video Summary", interactive=False)
        
        with gr.Row():
            analyze_btn = gr.Button("Analyze Video", variant="primary")
    
    with gr.Tab("Q&A"):
        question_input = gr.Textbox(label="Ask about the video")
        answer_output = gr.Textbox(label="Answer")
        ask_btn = gr.Button("Get Answer", variant="secondary")

    def analyze_video(url):
        transcript = get_transcript(url)
        return generate_summary(transcript)
    
    def handle_question(url, question):
        transcript = get_transcript(url)
        return answer_question(transcript, question)

    analyze_btn.click(
        fn=analyze_video,
        inputs=url_input,
        outputs=summary_output
    )
    
    ask_btn.click(
        fn=handle_question,
        inputs=[url_input, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    app.launch()