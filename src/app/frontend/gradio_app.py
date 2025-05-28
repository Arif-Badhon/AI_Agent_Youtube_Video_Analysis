import gradio as gr
gr.set_static_paths(paths=["assests"])
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.backend.agent import generate_summary, generate_mindmap, qa_agent
from app.backend.utils import get_transcript, get_video_metadata

import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Futuristic dark theme with neon accents
custom_theme = gr.themes.Default(
    primary_hue="blue", 
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    button_primary_background_fill="#00D4FF",
    button_primary_text_color="#000",
    button_primary_background_fill_hover="#0099CC",
    button_secondary_background_fill="rgba(0, 212, 255, 0.1)",
    button_secondary_text_color="#00D4FF",
    button_secondary_background_fill_hover="rgba(0, 212, 255, 0.2)",
    block_background_fill="#0A0A0F",
    background_fill_primary="#050508",
    background_fill_secondary="#0F0F15",
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

.gradio-container {
    max-width: 100vw !important; 
    margin: 0 !important;
    background: linear-gradient(135deg, #050508 0%, #0A0A0F 50%, #0F0F15 100%) !important;
    min-height: 100vh;
}

footer {visibility: hidden}

/* Neon title styling */
.neon-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 4rem !important;
    font-weight: 900 !important;
    text-align: center !important;
    color: #00D4FF !important;
    text-shadow: 
        0 0 5px #00D4FF,
        0 0 10px #00D4FF,
        0 0 20px #00D4FF,
        0 0 40px #00D4FF,
        0 0 80px #00D4FF !important;
    letter-spacing: 0.1em !important;
    margin: 2rem 0 !important;
    animation: pulse-glow 2s ease-in-out infinite alternate;
}

@keyframes pulse-glow {
    from {
        text-shadow: 
            0 0 5px #00D4FF,
            0 0 10px #00D4FF,
            0 0 20px #00D4FF,
            0 0 40px #00D4FF;
    }
    to {
        text-shadow: 
            0 0 10px #00D4FF,
            0 0 20px #00D4FF,
            0 0 30px #00D4FF,
            0 0 60px #00D4FF,
            0 0 100px #00D4FF;
    }
}

/* Main layout grid */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

/* Neon card styling */
.neon-card {
    background: rgba(15, 15, 21, 0.8) !important;
    border: 2px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    box-shadow: 
        0 0 20px rgba(0, 212, 255, 0.1),
        inset 0 0 20px rgba(0, 212, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
}

.neon-card:hover {
    border-color: rgba(0, 212, 255, 0.6) !important;
    box-shadow: 
        0 0 30px rgba(0, 212, 255, 0.2),
        inset 0 0 30px rgba(0, 212, 255, 0.1) !important;
    transform: translateY(-5px) !important;
}

/* Input section styling */
.input-section {
    grid-column: 1 / -1;
    margin-bottom: 2rem;
}

.neon-input {
    background: rgba(0, 0, 0, 0.5) !important;
    border: 2px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: 15px !important;
    color: #00D4FF !important;
    font-family: 'Exo 2', sans-serif !important;
    padding: 1rem !important;
}

.neon-input:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
}

/* Video preview styling */
.video-preview {
    position: relative;
    overflow: hidden;
}

.video-preview img {
    border-radius: 15px !important;
    width: 100% !important;
    height: auto !important;
    border: 2px solid rgba(0, 212, 255, 0.3) !important;
}

/* Summary section styling */
.summary-section {
    font-family: 'Exo 2', sans-serif !important;
}

#summary_output, #translated_output {
    background: rgba(0, 0, 0, 0.6) !important;
    color: #E0E0FF !important;
    padding: 1.5rem !important;
    border-radius: 15px !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    font-family: 'Exo 2', sans-serif !important;
    line-height: 1.6 !important;
}

/* Mind map styling */
.mindmap-container {
    grid-column: 1 / -1;
    margin-top: 2rem;
}

#mindmap-output img {
    border-radius: 15px !important;
    border: 2px solid rgba(0, 212, 255, 0.3) !important;
    background: rgba(0, 0, 0, 0.3) !important;
    width: 100% !important;
    height: auto !important;
}

/* Q&A section styling */
.qa-section {
    grid-column: 1 / -1;
    margin-top: 2rem;
}

.chatbot {
    background: rgba(0, 0, 0, 0.4) !important;
    border: 2px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: 20px !important;
    min-height: 400px !important;
}

/* Button styling */
.neon-button {
    background: linear-gradient(45deg, #00D4FF, #0099CC) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 24px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
    transition: all 0.3s ease !important;
}

.neon-button:hover {
    background: linear-gradient(45deg, #33E5FF, #00BBEE) !important;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.5) !important;
    transform: scale(1.05) !important;
}

/* Suggested questions styling */
.suggested-question {
    background: rgba(0, 212, 255, 0.1) !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    color: #00D4FF !important;
    border-radius: 15px !important;
    padding: 10px 15px !important;
    margin: 5px !important;
    font-family: 'Exo 2', sans-serif !important;
    transition: all 0.3s ease !important;
}

.suggested-question:hover {
    background: rgba(0, 212, 255, 0.2) !important;
    border-color: #00D4FF !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.3) !important;
}

/* Accordion styling */
.accordion {
    background: rgba(15, 15, 21, 0.6) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 15px !important;
    margin: 1rem 0 !important;
}

/* Copy button styling */
.copy-btn {
    background: rgba(0, 212, 255, 0.2) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0, 212, 255, 0.4) !important;
    border-radius: 10px !important;
    font-family: 'Exo 2', sans-serif !important;
}

.copy-btn:hover {
    background: rgba(0, 212, 255, 0.3) !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.3) !important;
}

/* Section headers */
.section-header {
    font-family: 'Orbitron', monospace !important;
    color: #00D4FF !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-grid {
        grid-template-columns: 1fr;
        padding: 1rem;
    }
    
    .neon-title {
        font-size: 2.5rem !important;
    }
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.3);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(45deg, #00D4FF, #0099CC);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(45deg, #33E5FF, #00BBEE);
}
"""

def analyze_video(url, mode):
    try:
        transcript = get_transcript(url)
        qa_agent.process_transcript(url, transcript)
        if not transcript:
            raise ValueError("Transcript is empty")
        summary = generate_summary(transcript, mode)
        mindmap = generate_mindmap(summary)
        return (f"### {mode.capitalize()} Summary\n\n{summary}", mindmap, "")
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
            {"role": "assistant", "content": answer},
        ]
    except Exception as e:
        return history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"⚠️ Error: {str(e)}"},
        ]

def update_suggested_questions(summary):
    """Generate questions and return visibility state"""
    if not summary or "Error:" in summary:
        return [gr.update(visible=False)]*3 + [gr.Row.update(visible=False)]
    
    try:
        questions = qa_agent.generate_questions(summary, num_questions=3)
        return [
            gr.update(visible=True, value=questions[0]),
            gr.update(visible=len(questions)>1, value=questions[1] if len(questions)>1 else ""),
            gr.update(visible=len(questions)>2, value=questions[2] if len(questions)>2 else ""),
            gr.Row.update(visible=len(questions)>0)
        ]
    except Exception as e:
        print(f"Question generation error: {e}")
        return [gr.update(visible=False)]*3 + [gr.Row.update(visible=False)]

def update_preview(url):
    if not url or not url.startswith(("https://www.youtube.com", "https://youtu.be")):
        return [None, "", "", False]
    
    metadata = get_video_metadata(url)
    if not metadata:
        return [
            None,
            "### Video Not Found",
            "The video may be private or unavailable",
            True
        ]
    
    return [
        metadata["thumbnail_url"],
        f"### {metadata['title']}",
        f"**Channel**: {metadata['channel']}",
        True
    ]

def update_row_visibility(should_show):
    return gr.Row.update(visible=should_show)

def use_suggested_question(question):
    """Insert question into input and clear suggestions"""
    return (
        question,
        *[gr.update(visible=False) for _ in range(3)],
        gr.Row.update(visible=False)
    )

with gr.Blocks(theme=custom_theme, css=css, title="Hermes AI") as app:
    
    # ========== Neon Header ==========
    gr.HTML(
    '''
    <div style="display: flex; align-items: center; justify-content: center; gap: 1.2rem; margin-top: 2rem; margin-bottom: 2rem;">
        <img src="/gradio_api/file=assests/hermes.png" alt="Logo" style="height: 64px; width: 64px; object-fit: contain;">
        <span class="neon-title" style="margin: 0;">HERMES AI</span>
    </div>
    '''
            )    
    # ========== Input Section ==========
    with gr.Row(elem_classes="input-section"):
        with gr.Column(elem_classes="neon-card"):
            gr.HTML('<h2 class="section-header">Video Analysis</h2>')
            with gr.Row():
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://youtube.com/watch?v=...",
                    max_lines=1,
                    scale=3,
                    elem_classes="neon-input"
                )
                mode_selector = gr.Radio(
                    choices=["Short", "Medium", "Detailed"],
                    value="Medium",
                    label="Summary Mode",
                    scale=1,
                )
            analyze_btn = gr.Button(
                "🚀 Analyze Video", 
                variant="primary", 
                elem_classes="neon-button",
                size="lg"
            )
    
    # ========== Video Preview (Hidden by default) ==========
    with gr.Row(visible=False, elem_classes="neon-card") as preview_row:
        with gr.Column(scale=1, elem_classes="video-preview"):
            thumbnail = gr.Image(label="Video Preview", show_label=False)
        with gr.Column(scale=2):
            video_title = gr.Markdown()
            video_channel = gr.Markdown()
    
    visibility_tracker = gr.Textbox(visible=False)
    
    # ========== Main Content Grid ==========
    with gr.Row():
        # Left Column - Video & Mind Map
        with gr.Column(scale=1):
            # Summary Section
            with gr.Column(elem_classes="neon-card summary-section"):
                gr.HTML('<h2 class="section-header">Summary</h2>')
                summary_output = gr.Markdown(elem_id="summary_output")
                copy_summary_btn = gr.Button("📋 Copy Summary", elem_classes="copy-btn")
        
        # Right Column - Translation & Q&A
        with gr.Column(scale=1):
            # Translation Section
            with gr.Column(elem_classes="neon-card"):
                gr.HTML('<h2 class="section-header">🌍 Translation</h2>')
                with gr.Row():
                    lang_selector = gr.Dropdown(
                        choices=[
                            "English", "Bengali", "Hindi", "Chinese", 
                            "Spanish", "French", "Arabic", "Russian"
                        ],
                        value="English",
                        label="Target Language",
                        scale=2
                    )
                    translate_btn = gr.Button("Translate", elem_classes="neon-button", scale=1)
                translated_output = gr.Markdown(elem_id="translated_output")
                copy_translate_btn = gr.Button("📋 Copy Translation", elem_classes="copy-btn")
    
    # ========== Mind Map Section ==========
    with gr.Row(elem_classes="mindmap-container"):
        with gr.Column(elem_classes="neon-card"):
            gr.HTML('<h2 class="section-header">🧠 Interactive Mind Map</h2>')
            mindmap_output = gr.Image(
                label="Concept Visualization",
                elem_id="mindmap-output",
                show_download_button=True,
                show_label=False
            )
    
    # ========== Q&A Section ==========
    with gr.Row(elem_classes="qa-section"):
        with gr.Column(elem_classes="neon-card"):
            gr.HTML('<h2 class="section-header">💬 AI-Powered Q&A</h2>')
            qa_chat = gr.Chatbot(
                height=400,
                avatar_images=(
                    "https://i.imgur.com/7kQEsHU.png",
                    "https://i.imgur.com/8EeSUQ3.png",
                ),
                show_label=False,
                type="messages",
                elem_classes="chatbot"
            )
            
            # Suggested Questions
            with gr.Row(visible=False) as suggested_questions_row:
                suggested_btns = [
                    gr.Button(visible=False, elem_classes="suggested-question") 
                    for _ in range(3)
                ]
            
            # Input Section
            with gr.Row():
                qa_input = gr.Textbox(
                    placeholder="Ask anything about the video...",
                    show_label=False,
                    scale=4,
                    elem_classes="neon-input"
                )
                qa_btn = gr.Button("Ask AI", elem_classes="neon-button", scale=1)

    # ========== Event Handling ==========
    url_input.input(
        fn=update_preview,
        inputs=url_input,
        outputs=[thumbnail, video_title, video_channel, visibility_tracker]
    )

    visibility_tracker.change(
        fn=update_row_visibility,
        inputs=visibility_tracker,
        outputs=preview_row
    )

    analyze_btn.click(
        analyze_video,
        inputs=[url_input, mode_selector],
        outputs=[summary_output, mindmap_output, translated_output]
    ).then(
        update_suggested_questions,
        inputs=summary_output,
        outputs=[*suggested_btns, suggested_questions_row]
    )

    for btn in suggested_btns:
        btn.click(
            fn=use_suggested_question,
            inputs=btn,
            outputs=[qa_input, *suggested_btns, suggested_questions_row]
        ).then(
            fn=handle_qa,
            inputs=[qa_chat, url_input, qa_input],
            outputs=qa_chat
        )

    translate_btn.click(
        handle_translation,
        inputs=[summary_output, lang_selector],
        outputs=translated_output,
    )

    qa_btn.click(handle_qa, inputs=[qa_chat, url_input, qa_input], outputs=qa_chat)
    qa_input.submit(handle_qa, inputs=[qa_chat, url_input, qa_input], outputs=qa_chat)

    # Copy functionality
    copy_summary_btn.click(
        None,
        inputs=summary_output,
        outputs=None,
        js="async (text) => { await navigator.clipboard.writeText(text); }",
    )
    copy_translate_btn.click(
        None,
        inputs=translated_output,
        outputs=None,
        js="async (text) => { await navigator.clipboard.writeText(text); }",
    )

if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", allowed_paths=["assests"])