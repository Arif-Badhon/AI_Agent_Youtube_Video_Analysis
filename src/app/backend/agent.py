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
        """Improved context-preserving chunking"""
        paragraphs = re.split(r'\n+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para.split())
            if current_length + para_length <= chunk_size:
                current_chunk.append(para)
                current_length += para_length
            else:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
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
        top_k = min(5, len(similarities))  # Dynamic chunk selection
        top_idxs = np.argpartition(similarities, -top_k)[-top_k:]
        context = " ".join([data['chunks'][i] for i in top_idxs])
        
        result = self.qa_model(
            question=question,
            context=context,
            max_answer_len=150,
            handle_impossible_answer=True
        )
        
        if result['answer'] == "" or result['score'] < 0.1:
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

def generate_summary(transcript, mode):
    """Your summary logic (replace with actual implementation)"""
    return f"{mode.capitalize()} summary: {transcript[:500]}..."

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
