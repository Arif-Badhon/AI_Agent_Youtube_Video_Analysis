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
    body_text_color_subdued="black"
)

css = """
.gradio-container {
    max-width: 100vw !important;
    width: 100vw !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
footer {visibility: hidden}
@import url('https://fonts.boomla.com/bangla.css');
body {font-family: 'SolaimanLipi', 'Segoe UI', Arial, sans-serif!important;}
#main-card {
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 4px 28px #0001;
    padding: 30px 32px 32px 32px;
    margin-top: 40px;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}
.section-title {font-size: 1.2em; font-weight: 600; margin-bottom: 10px; color: #2c3e50;}
#summary_output, #translated_output {
    background: #111;
    color: #fff;
    border-radius: 10px;
    padding: 18px;
    min-height: 120px;
    font-size: 1.08em;
    border: 1.5px solid #1976d2;
    box-shadow: 0 2px 8px #0001;
    word-break: break-word;
}
#mindmap_card {background: #f3f7fb; border-radius: 10px; padding: 18px;}
#detected_lang {font-size: 0.95em; color: #888; margin-bottom: 10px;}
.copy-btn {
    background: #1976d2!important;
    color: #fff!important;
    border: none!important;
    border-radius: 8px!important;
    padding: 7px 18px!important;
    margin-left: 8px!important;
    font-weight: 600!important;
    font-size: 1em!important;
    box-shadow: 0 2px 8px #0002;
    transition: background 0.2s;
}
.copy-btn:hover {
    background: #1565c0!important;
}
#footer {text-align: center; color: #aaa; margin-top: 38px; font-size: 0.93em;}
@media (max-width: 600px) {
  #main-card {padding: 10px;}
}
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

def analyze_video_handler(url, mode):
    if not (url.startswith("https://www.youtube.com/") or url.startswith("https://youtu.be/")):
        raise gr.Error("Invalid YouTube URL format")
    try:
        transcript = get_transcript(url)
        summary = generate_summary(transcript, mode.lower())
        mindmap_path = generate_mindmap(summary)
        summary_md = f"### {mode} Summary\n\n" + summary.replace('\n', '\n\n')
        return summary_md, mindmap_path, ""
    except Exception as e:
        return f"**Analysis failed:** {str(e)}", None, ""

def translate_handler(summary_md, target_lang):
    import re
    summary = re.sub(r"^#+.*\n", "", summary_md)
    if not summary or summary.lower().startswith("**analysis failed"):
        raise gr.Error("Generate a valid summary first")
    lang_code = LANGUAGE_MAPPING.get(target_lang, "en_XX")
    try:
        translated = translate_summary(summary, lang_code)
        return f"### {target_lang} Translation\n\n{translated.replace(chr(10), chr(10)+chr(10))}"
    except Exception as e:
        return f"**Translation failed:** {str(e)}"

copy_js = """
async (text) => {
    try {
        await navigator.clipboard.writeText(text);
        return "Copied!";
    } catch (e) {
        return "Copy failed";
    }
}
"""

with gr.Blocks(theme=custom_theme, css=css) as app:
    gr.Markdown("# YouTube AI Analyzer 3.0 🚀")
    with gr.Column(elem_id="main-card"):
        with gr.Tab("Video Analysis"):
            gr.Markdown('<div class="section-title">Step 1: Paste YouTube Link</div>')
            url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="Paste video URL here...",
                info="Supports both https://www.youtube.com/ and https://youtu.be/ links."
            )
            with gr.Row():
                mode_dropdown = gr.Radio(
                    choices=["Short", "Medium", "Detailed"],
                    value="Medium",
                    label="Summary Length",
                    interactive=True
                )
                analyze_btn = gr.Button("🔍 Analyze Video", variant="primary")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown('<div class="section-title">Summary</div>')
                    detected_lang = gr.Markdown("", elem_id="detected_lang")
                    summary_output = gr.Markdown(elem_id="summary_output")
                    with gr.Row():
                        copy_summary_btn = gr.Button("Copy", elem_id="copy_summary", elem_classes="copy-btn")
                with gr.Column(scale=1):
                    gr.Markdown('<div class="section-title">Mind Map</div>')
                    with gr.Column(elem_id="mindmap_card"):
                        vis_output = gr.Image(
                            show_label=False,
                            height=220,
                            width=320,
                            show_download_button=True
                        )
            with gr.Accordion("🌐 Translate Summary", open=False):
                with gr.Row():
                    lang_dropdown = gr.Dropdown(
                        choices=list(LANGUAGE_MAPPING.keys()),
                        value="English",
                        label="Target Language",
                        interactive=True
                    )
                    translate_btn = gr.Button("🌐 Translate")
                translated_output = gr.Markdown(elem_id="translated_output")
                with gr.Row():
                    copy_translated_btn = gr.Button("Copy", elem_id="copy_translated", elem_classes="copy-btn")
        gr.Markdown(
            """
            <div id="footer">
            <b>Tips:</b> Click "Copy" to copy summaries. Download the mind map for your notes.<br>
            <i>Powered by Gradio, Transformers, and mBART.</i>
            </div>
            """
        )

    analyze_btn.click(
        fn=analyze_video_handler,
        inputs=[url_input, mode_dropdown],
        outputs=[summary_output, vis_output, detected_lang],
        show_progress="full"
    )
    translate_btn.click(
        fn=translate_handler,
        inputs=[summary_output, lang_dropdown],
        outputs=translated_output
    )
    copy_summary_btn.click(None, inputs=summary_output, outputs=None, js=copy_js)
    copy_translated_btn.click(None, inputs=translated_output, outputs=None, js=copy_js)

if __name__ == "__main__":
    app.launch()
