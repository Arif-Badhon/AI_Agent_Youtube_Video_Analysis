import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.backend.agent import generate_summary, answer_question, translate_summary
from app.backend.utils import *

css = """
.gradio-container {max-width: 900px!important}
footer {visibility: hidden}
@import url('https://fonts.boomla.com/bangla.css');
body {font-family: 'SolaimanLipi', sans-serif!important}
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

with gr.Blocks(theme=gr.themes.Soft(), css=css) as app:
    gr.Markdown("# YouTube AI Analyzer 2.0 🚀")
    
    with gr.Tab("Video Analysis"):
        with gr.Row():
            url_input = gr.Textbox(label="YouTube URL", placeholder="Paste video URL here...")
            summary_output = gr.Textbox(label="Video Summary", lines=5, interactive=False)
        
        with gr.Row():
            analyze_btn = gr.Button("Analyze Video", variant="primary")
            
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

    def analyze_video_handler(url):
        if not url.startswith("https://www.youtube.com/"):
            raise gr.Error("Invalid YouTube URL format")
        
        try:
            transcript = get_transcript(url)
            return generate_summary(transcript)
        except Exception as e:
            return f"Analysis failed: {str(e)}\n\nNote: Could not find captions and audio transcription failed."

    def translate_handler(summary, target_lang):
        if not summary:
            raise gr.Error("Generate a summary first before translating")
        
        lang_code = LANGUAGE_MAPPING.get(target_lang, "en_XX")
        try:
            return translate_summary(summary, lang_code)
        except Exception as e:
            return f"Translation failed: {str(e)}"

    # Event bindings
    analyze_btn.click(
        fn=analyze_video_handler,
        inputs=url_input,
        outputs=summary_output
    )
    
    translate_btn.click(
        fn=translate_handler,
        inputs=[summary_output, lang_dropdown],
        outputs=translated_output
    )

if __name__ == "__main__":
    app.launch()
