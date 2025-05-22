from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt
import torch

class QAAgent:
    def __init__(self):
        # QA System
        self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-large-squad2")
        qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-large-squad2")
        self.qa_model = pipeline(
            "question-answering",
            model=qa_model,
            tokenizer=qa_tokenizer
        )
        self.cache = {}

        # Translation System
        self.translator = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.trans_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.lang_map = {
            "English": "en_XX",
            "Bengali": "bn_IN",
            "Hindi": "hi_IN",
            "Chinese": "zh_CN",
            "Spanish": "es_XX",
            "French": "fr_XX",
            "Arabic": "ar_AR",
            "Russian": "ru_RU"
        }

    def chunk_text(self, text, chunk_size=500):
        """Enhanced chunking with overlap"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i in range(len(words)):
            current_chunk.append(words[i])
            current_length += 1
            
            if current_length >= chunk_size:
                chunks.append(' '.join(current_chunk))
                # Keep 20% overlap
                current_chunk = current_chunk[-int(chunk_size*0.2):]
                current_length = len(current_chunk)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks


    def process_transcript(self, url, transcript):
        video_id = url.split("v=")[-1]
        if video_id not in self.cache:
            chunks = self.chunk_text(transcript)
            self.cache[video_id] = {
                'chunks': chunks,
                'embeddings': self.embedder.encode(chunks)
            }

    def answer_question(self, url, question):
        video_id = url.split("v=")[-1]
        if video_id not in self.cache:
            raise ValueError("Process transcript first")
            
        data = self.cache[video_id]
        question_embed = self.embedder.encode([question])
        
        similarities = cosine_similarity(question_embed, data['embeddings'])[0]
        
        # Handle case with few chunks
        valid_k = min(3, len(similarities))
        if valid_k == 0:
            return "❌ Not enough context to answer this question"
            
        top_idxs = np.argsort(similarities)[-valid_k:]  # Use argsort instead of argpartition
        context = " ".join([data['chunks'][i] for i in top_idxs])
        
        result = self.qa_model(
            question=question,
            context=context,
            max_answer_len=150,
            handle_impossible_answer=True
        )
        
        if result['answer'] == "" or result['score'] < 0.15:  # Slightly lower threshold
            return "🔍 This information isn't clearly covered in the video."
        return f"{result['answer']} (Confidence: {result['score']:.0%})"


    def translate_text(self, text, target_lang):
        if target_lang not in self.lang_map:
            raise ValueError(f"Unsupported language: {target_lang}")
            
        self.trans_tokenizer.src_lang = "en_XX"
        encoded = self.trans_tokenizer(text, return_tensors="pt")
        generated_tokens = self.translator.generate(
            **encoded,
            forced_bos_token_id=self.trans_tokenizer.lang_code_to_id[self.lang_map[target_lang]]
        )
        return self.trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn",
    device=0 if torch.cuda.is_available() else -1
)
def chunk_text(text, max_tokens=900):
    """Split text into roughly max_tokens word chunks."""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i+max_tokens])

def generate_summary(transcript, mode):
    # Set summary lengths based on mode
    if mode.lower() == "short":
        max_length, min_length = 60, 20
    elif mode.lower() == "medium":
        max_length, min_length = 120, 40
    else:  # detailed
        max_length, min_length = 200, 60

    # Always chunk transcript if too long
    summaries = []
    for chunk in chunk_text(transcript, max_tokens=900):  # 900 is safe for BART
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        summaries.append(summary)
    # If multiple chunks, summarize the summaries
    if len(summaries) > 1:
        combined = " ".join(summaries)
        final_summary = summarizer(
            combined,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        return final_summary
    else:
        return summaries[0]

def generate_mindmap(summary):
    """Generate mindmap visualization"""
    words = [word for word in summary.split() if len(word) > 3][:15]
    G = nx.Graph()
    for i, word in enumerate(words):
        G.add_node(word)
        if i > 0:
            G.add_edge(words[i-1], word)
    
    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(
        G, pos, with_labels=True, 
        node_size=2500, node_color="skyblue",
        font_size=10, font_weight="bold",
        edge_color="gray"
    )
    plt.savefig("mindmap.png", bbox_inches="tight")
    plt.close()
    return "mindmap.png"

qa_agent = QAAgent()
