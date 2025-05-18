import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.backend.agent import generate_summary, answer_question, translate_summary, generate_mindmap
from app.backend.utils import *

custom_theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray"
).set(
    body_background_fill="white",
    block_background_fill="white",
    button_primary_background_fill="#000000",
    button_primary_text_color="white",
    button_primary_background_fill_hover="#1a1a1a",
    block_label_background_fill="#000000",
    block_label_text_color="white",
    # Valid text color parameter:
    body_text_color_subdued="black"
)


css = """
.gradio-container {max-width: 900px!important}
footer {visibility: hidden}
@import url('https://fonts.boomla.com/bangla.css');
body {font-family: 'SolaimanLipi', sans-serif!important}
.dark .summary-box textarea {background-color: #1a1a1a!important}
.md_output {margin-top: 10px!important}
"""

LANGUAGE_MAPPING = {
    "English": "en_XX",
    "French": "fr_XX",
    "Spanish": "es_XX", 
    "German": "de_DE",
    "Chinese": "zh_CN",
    "Hindi": "hi_IN",
    "Arabic": "ar_AR",
    "Russian": "ru_RU",
    "Bengali": "bn_IN"
}

with gr.Blocks(theme=custom_theme, css=css) as app:
    gr.Markdown("# YouTube AI Analyzer 3.0 🚀")
    
    with gr.Tab("Video Analysis"):
        with gr.Row():
            url_input = gr.Textbox(label="YouTube URL", placeholder="Paste video URL here...")
            summary_output = gr.Textbox(label="Video Summary", lines=5, interactive=False)
        
        with gr.Row():
            with gr.Column(scale=2):
                mode_dropdown = gr.Dropdown(
                    choices=["Short", "Medium", "Detailed"],
                    value="Medium",
                    label="Summary Length",
                    interactive=True
                )
        with gr.Row():
            analyze_btn = gr.Button("Analyze Video", variant="primary")
            vis_output = gr.Image(label="Visual Summary (Mind Map)", show_label=True)
            
        with gr.Accordion("Translation Settings", open=True):
            with gr.Row():
                lang_dropdown = gr.Dropdown(
                    choices=list(LANGUAGE_MAPPING.keys()),
                    value="English",
                    label="Target Language",
                    interactive=True
                )
                translate_btn = gr.Button("Translate Summary", variant="secondary")
            
            translated_output = gr.Textbox(label="Translated Summary", lines=5, interactive=False)

    def analyze_video_handler(url, mode):
        if not url.startswith("https://www.youtube.com/"):
            raise gr.Error("Invalid YouTube URL format")
        
        try:
            transcript = get_transcript(url)
            summary = generate_summary(transcript, mode.lower())
            mindmap_path = generate_mindmap(summary)
            return summary, mindmap_path  # Return both values
        except Exception as e:
            return f"Analysis failed: {str(e)}", None  # Return None for image


    def translate_handler(summary, target_lang):
        if not summary or summary.startswith("Analysis failed"):
            raise gr.Error("Generate a valid summary first")
        
        lang_code = LANGUAGE_MAPPING.get(target_lang, "en_XX")
        try:
            return translate_summary(summary, lang_code)
        except Exception as e:
            return f"Translation failed: {str(e)}"

    # Event bindings
    analyze_btn.click(
        fn=analyze_video_handler,
        inputs=[url_input, mode_dropdown],
        outputs=[summary_output, vis_output],
        show_progress="full"
    )
    
    translate_btn.click(
        fn=translate_handler,
        inputs=[summary_output, lang_dropdown],
        outputs=translated_output
    )

if __name__ == "__main__":
    app.launch()
