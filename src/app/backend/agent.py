# backend/agent.py
import os
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSeq2SeqLM
import torch
import whisper
from typing import Tuple
from transformers import pipeline

# Load environment variables
load_dotenv()

#client = OpenAI(OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"))

#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = os.getenv("MODEL_NAME")  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=1024)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt"
    # Removed truncation and max_length here
)



summary_template = """Generate concise video summary with timestamps:
{transcript}
"""

qa_template = """Answer using video transcript:
Question: {question}
Transcript: {transcript}
"""
WHISPER_MODEL = "base"  # Change to "small", "medium", or "large" as needed

# Add this function
def get_whisper_model():
    return whisper.load_model(WHISPER_MODEL)

def truncate_text_to_1024_tokens(text, tokenizer):
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) > 1024:
        tokens = tokens[:1024]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def chunk_text(text, max_tokens=1000):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def generate_summary(transcript, mode="medium"):
    valid_modes = ["short", "medium", "detailed"]
    mode = mode.lower()
    if mode not in valid_modes:
        mode = "medium"
    
    mode_settings = {
        "short": {"max": 150, "min": 40, "beams": 4},
        "medium": {"max": 300, "min": 120, "beams": 5},
        "detailed": {"max": 600, "min": 200, "beams": 6}
    }
    
    config = mode_settings[mode]
    truncated_transcript = truncate_text_to_1024_tokens(transcript, tokenizer)
    
    return summarizer(
        truncated_transcript,
        max_length=config["max"],
        min_length=config["min"],
        num_beams=config["beams"],
        no_repeat_ngram_size=3,
        do_sample=False,
        truncation=True
    )[0]['summary_text']




qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

def answer_question(transcript, question):
    return qa_pipeline(
        question=question,
        context=transcript[:1024]  # Model's max context
    )['answer']


# agent.py - Add translation functionality
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translation_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

LANGUAGE_MAPPING = {
    "French": "fr_XX",
    "Spanish": "es_XX",
    "German": "de_DE",
    "Chinese": "zh_CN",
    "Hindi": "hi_IN",
    "Bengali": "bn_IN"
}

def translate_summary(summary, target_lang):
    translation_tokenizer.src_lang = "en_XX"
    inputs = translation_tokenizer(
        summary, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    )
    
    generated_tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=translation_tokenizer.lang_code_to_id[target_lang]
    )
    
    return translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

from app.backend.visualizer import VisualSummarizer

visualizer = VisualSummarizer()

def generate_mindmap(transcript):
    try:
        return visualizer.create_mindmap(transcript)
    except Exception as e:
        return f"Mindmap generation failed: {str(e)}"

