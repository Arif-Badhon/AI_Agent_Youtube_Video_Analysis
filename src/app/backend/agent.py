# backend/agent.py
#from openai import OpenAI
#from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from transformers import pipeline
import torch

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
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)



summary_template = """Generate concise video summary with timestamps:
{transcript}
"""

qa_template = """Answer using video transcript:
Question: {question}
Transcript: {transcript}
"""

def generate_summary(transcript):
    return summarizer(
        transcript,
        max_length=150,
        min_length=40,
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
