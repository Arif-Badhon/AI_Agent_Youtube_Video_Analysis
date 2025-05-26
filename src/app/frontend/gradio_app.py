import gradio as gr
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.backend.agent import generate_summary, generate_mindmap, qa_agent
from app.backend.utils import get_transcript, get_video_metadata

# Add at the top of your gradio_app.py
import warnings
from transformers import logging
logging.set_verbosity_error()  # Suppress all transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)  # General warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

custom_theme = gr.themes.Default(primary_hue="blue", secondary_hue="slate").set(
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
button.suggested-question {
    max-width: 300px;
    white-space: normal;
    line-height: 1.2;
    padding: 8px 12px;
    margin: 4px;
    flex-grow: 1;
    text-align: left;
    height: auto;
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

with gr.Blocks(theme=custom_theme, css=css) as app:
    # ========== Header with Logo ==========
    gr.Image('assests/hermes.png', show_label=False, width=60, height=100, elem_id="logo")
    gr.Markdown('<h1 style="text-align:center;">Hermes AI</h1>')

    # ========== Metadata Preview ==========
    with gr.Row(variant="panel", visible=False) as preview_row:
        thumbnail = gr.Image(label="Video Thumbnail", width=200)
        with gr.Column():
            video_title = gr.Markdown()
            video_channel = gr.Markdown()

    # Hidden component to control visibility
    visibility_tracker = gr.Textbox(visible=False)
    visibility_state = gr.State(False)

    # ========== Main Card ==========
    with gr.Column(elem_id="main-card"):
        # Input Section
        with gr.Row():
            url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="Paste video link here...",
                max_lines=1,
                scale=4,
            )
            mode_selector = gr.Radio(
                choices=["Short", "Medium", "Detailed"],
                value="Medium",
                label="Summary Mode",
                scale=2,
            )
            analyze_btn = gr.Button("Analyze Video", variant="primary", scale=1)

        # Summary Section
        summary_output = gr.Markdown(elem_id="summary_output")
        copy_summary_btn = gr.Button("📋 Copy Summary", elem_classes="copy-btn")

        # Translation Section
        with gr.Accordion("🌍 Translation (Supports 8 Languages)", open=False):
            with gr.Row():
                lang_selector = gr.Dropdown(
                    choices=[
                        "English",
                        "Bengali",
                        "Hindi",
                        "Chinese",
                        "Spanish",
                        "French",
                        "Arabic",
                        "Russian",
                    ],
                    value="English",
                    label="Target Language",
                    scale=3,
                )
                translate_btn = gr.Button("Translate", scale=1)
            translated_output = gr.Markdown(elem_id="translated_output")
            copy_translate_btn = gr.Button(
                "📋 Copy Translation", elem_classes="copy-btn"
            )

        # Mind Map
        with gr.Accordion("🧠 Interactive Mind Map", open=True):
            mindmap_output = gr.Image(
                label="Concept Visualization",
                elem_id="mindmap-output",
                show_download_button=True,
            )

        # Q&A Section
        with gr.Accordion("💬 AI-Powered Q&A", open=False):
            qa_chat = gr.Chatbot(
                height=400,
                avatar_images=(
                    "https://i.imgur.com/7kQEsHU.png",
                    "https://i.imgur.com/8EeSUQ3.png",
                ),
                show_label=False,
                type="messages",
            )
            
            # Suggested Questions
            suggested_questions_visible = gr.State(False)
            with gr.Row(visible=False) as suggested_questions_row:
                gr.Markdown("**Suggested Questions:**")
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
                )
                qa_btn = gr.Button("Ask AI", variant="secondary", scale=1)

    # ========== Metadata Preview Logic ==========
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

    url_input.input(
        fn=update_preview,
        inputs=url_input,
        outputs=[thumbnail, video_title, video_channel, visibility_tracker]
    )

    # Replace visibility_tracker logic with:
    def update_row_visibility(should_show):
        return gr.Row.update(visible=should_show)

    visibility_tracker.change(
        fn=update_row_visibility,
        inputs=visibility_tracker,
        outputs=preview_row
    )


    # ========== Suggested Questions Logic ==========
    def update_suggested_questions(summary):
        """Generate questions and return visibility state"""
        if not summary or "Error:" in summary:
            return [{"visible": False}]*3 + [False]
        
        try:
            # Your question generation logic here
            questions = ["Q1", "Q2", "Q3"]  # Replace with actual generated questions
            updates = []
            for i in range(3):
                updates.append({
                    "visible": i < len(questions),
                    "value": questions[i] if i < len(questions) else "",
                    "__type__": "update"
                })
            return updates + [len(questions) > 0]
        except Exception as e:
            return [{"visible": False}]*3 + [False]

    def update_row_visibility(visible):
        """Convert boolean to proper row update syntax"""
        return {"visible": visible, "__type__": "update"}

    # In your event handling:
    analyze_btn.click(
        analyze_video,
        inputs=[url_input, mode_selector],
        outputs=[summary_output, mindmap_output, translated_output]
    ).then(
        update_suggested_questions,
        inputs=summary_output,
        outputs=[*suggested_btns, visibility_state]
    ).then(
        update_row_visibility,
        inputs=visibility_state,
        outputs=suggested_questions_row
    )


    def use_suggested_question(question):
        """Insert question into input and clear suggestions"""
        return (
            question,  # Update qa_input
            {"visible": False, "__type__": "update"},  # Button 1
            {"visible": False, "__type__": "update"},  # Button 2
            {"visible": False, "__type__": "update"},  # Button 3
            {"visible": False, "__type__": "update"}   # Row visibility
        )

# Update button click handlers
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

    # ========== Event Handling ==========
    # Replace current analyze_btn.click() calls with:
    def clear_suggestions():
        return [{"visible": False}]*3 + [False]
    analyze_btn.click(
    fn=clear_suggestions,
    outputs=[*suggested_btns, suggested_questions_row]
).then(
    analyze_video,
    inputs=[url_input, mode_selector],
    outputs=[summary_output, mindmap_output, translated_output]
).then(
    fn=update_suggested_questions,
    inputs=summary_output,
    outputs=[*suggested_btns, suggested_questions_row]
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
        js="async (text) => { await navigator.clipboard.writeText(text) }",
    )
    copy_translate_btn.click(
        None,
        inputs=translated_output,
        outputs=None,
        js="async (text) => { await navigator.clipboard.writeText(text) }",
    )

if __name__ == "__main__":
    app.launch()
