# backend/agent.py
#from openai import OpenAI
#from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from transformers import pipeline
import torch
import whisper

# Load environment variables
load_dotenv()

#client = OpenAI(OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"))

#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = os.getenv("MODEL_NAME")  # or your chosen model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    framework="pt"
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


def generate_summary(transcript):
    return summarizer(
        transcript,
        max_length=300,  # Increased from 150
        min_length=120,   # Increased from 40
        num_beams=4,      # Better quality generation
        no_repeat_ngram_size=3,  # Prevent repetition
        do_sample=False,
        truncation=True,
        clean_up_tokenization_spaces=True
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
