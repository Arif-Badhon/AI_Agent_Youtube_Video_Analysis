import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.backend.agent import generate_summary, generate_mindmap, qa_agent
from app.backend.utils import get_transcript

# ========== Theme & CSS ==========
custom_theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate"
).set(
    button_primary_background_fill="#1976d2",
    button_primary_text_color="white",
    button_primary_background_fill_hover="#1565c0",
)

css = """
.gradio-container {max-width: 100vw!important; margin: 0!important}
footer {visibility: hidden}
#main-card {
    background: white; 
    border-radius: 16px;
    box-shadow: 0 4px 24px #0001;
    padding: 2.4rem;
    margin: 2rem auto;
    max-width: 900px;
}
#summary_output, #translated_output {
    background: #000 !important;
    color: #fff !important;
    padding: 20px;
    border-radius: 12px;
    margin: 15px 0;
}
#mindmap-output img {
    max-height: 70vh !important;
    width: 100% !important;
    object-fit: contain;
}
.copy-btn {
    background: #1976d2!important;
    color: white!important;
    margin: 10px 0;
}
.dark .copy-btn {
    background: #1565c0!important;
}
"""

# ========== Handlers ==========
def analyze_video(url, mode):
    try:
        transcript = get_transcript(url)
        qa_agent.process_transcript(url, transcript)
        
        if not transcript:
            raise ValueError("Transcript is empty")
            
        summary = generate_summary(transcript, mode)
        mindmap = generate_mindmap(summary)
        
        return (
            f"### {mode.capitalize()} Summary\n\n{summary}",
            mindmap,
            ""
        )
    except Exception as e:
        return f"**Error:** {str(e)}", None, ""

def handle_translation(summary, lang):
    try:
        translated = qa_agent.translate_text(summary, lang)
        return f"### {lang} Translation\n\n{translated}"
    except Exception as e:
        return f"**Translation Error:** {str(e)}"

def handle_qa(history, url, question):
    try:
        answer = qa_agent.answer_question(url, question)
        return history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    except Exception as e:
        return history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"⚠️ Error: {str(e)}"}
        ]


# ========== Interface ==========
with gr.Blocks(theme=custom_theme, css=css) as app:
    gr.Markdown("# 🎥 YouTube AI Analyzer 6.0")
    
    with gr.Column(elem_id="main-card"):
        # Input Section
        with gr.Row():
            url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="Paste video link here...",
                max_lines=1,
                scale=4
            )
            mode_selector = gr.Radio(
                choices=["Short", "Medium", "Detailed"],
                value="Medium",
                label="Summary Mode",
                scale=2
            )
            analyze_btn = gr.Button("Analyze Video", variant="primary", scale=1)
        
        # Summary Section
        summary_output = gr.Markdown(elem_id="summary_output")
        copy_summary_btn = gr.Button("📋 Copy Summary", elem_classes="copy-btn")
        
        # Translation Section
        with gr.Accordion("🌍 Translation (Supports 8 Languages)", open=False):
            with gr.Row():
                lang_selector = gr.Dropdown(
                    choices=["English", "Bengali", "Hindi", "Chinese",
                            "Spanish", "French", "Arabic", "Russian"],
                    value="English",
                    label="Target Language",
                    scale=3
                )
                translate_btn = gr.Button("Translate", scale=1)
            translated_output = gr.Markdown(elem_id="translated_output")
            copy_translate_btn = gr.Button("📋 Copy Translation", elem_classes="copy-btn")
        
        # Mind Map
        with gr.Accordion("🧠 Interactive Mind Map", open=True):
            mindmap_output = gr.Image(
                label="Concept Visualization",
                elem_id="mindmap-output",
                show_download_button=True
            )
        
        # Q&A Section
        with gr.Accordion("💬 AI-Powered Q&A", open=False):
            qa_chat = gr.Chatbot(
                height=400,
                avatar_images=(
                    "https://i.imgur.com/7kQEsHU.png",  # User avatar
                    "https://i.imgur.com/8EeSUQ3.png"   # Bot avatar
                ),
                show_label=False,
                type="messages"  # New format required
            )
            with gr.Row():
                qa_input = gr.Textbox(
                    placeholder="Ask anything about the video...",
                    show_label=False,
                    scale=4
                )
                qa_btn = gr.Button("Ask AI", variant="secondary", scale=1)

    # ========== Event Handling ==========
    analyze_btn.click(
        analyze_video,
        inputs=[url_input, mode_selector],
        outputs=[summary_output, mindmap_output, translated_output]
    )
    
    translate_btn.click(
        handle_translation,
        inputs=[summary_output, lang_selector],
        outputs=translated_output
    )
    
    qa_btn.click(
        handle_qa,
        inputs=[qa_chat, url_input, qa_input],
        outputs=qa_chat
    )
    qa_input.submit(
        handle_qa,
        inputs=[qa_chat, url_input, qa_input],
        outputs=qa_chat
    )
    
    # Copy functionality
    copy_summary_btn.click(
        None,
        inputs=summary_output,
        outputs=None,
        js="async (text) => { await navigator.clipboard.writeText(text) }"
    )
    copy_translate_btn.click(
        None,
        inputs=translated_output,
        outputs=None,
        js="async (text) => { await navigator.clipboard.writeText(text) }"
    )

if __name__ == "__main__":
    app.launch()
