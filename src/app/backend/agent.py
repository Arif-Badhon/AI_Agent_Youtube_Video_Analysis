from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    pipeline,
    AutoModelForSeq2SeqLM,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

# Global summarization components
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline(
    "summarization",
    model=summarizer_model,
    tokenizer=summarizer_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

def chunk_text(text, max_tokens=900):
    """Split text by tokens instead of words"""
    tokens = summarizer_tokenizer.encode(text, truncation=False, add_special_tokens=False)
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        yield summarizer_tokenizer.decode(chunk, skip_special_tokens=True)

def generate_summary(transcript, mode):
    """Global summary generation function"""
    if mode.lower() == "short":
        max_length, min_length = 40, 20
    elif mode.lower() == "medium":
        max_length, min_length = 100, 40
    else:  # detailed
        max_length, min_length = 150, 60

    summaries = []
    for chunk in chunk_text(transcript, max_tokens=900):
        inputs = summarizer_tokenizer(
            chunk, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        )
        truncated_chunk = summarizer_tokenizer.decode(inputs["input_ids"][0])
        summary = summarizer(
            truncated_chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )[0]["summary_text"]
        summaries.append(summary)

    if len(summaries) > 1:
        combined = " ".join(summaries)
        return summarizer(
            combined,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )[0]["summary_text"]
    return summaries[0] if summaries else "No summary generated"

class QAAgent:
    def __init__(self):
        # QA System
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "deepset/roberta-large-squad2"
        )
        qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-large-squad2")
        self.qa_model = pipeline(
            "question-answering",
            model=qa_model,
            tokenizer=qa_tokenizer,
            max_seq_len=512,
            device=0 if torch.cuda.is_available() else -1,
        )
        self.cache = {}

        # Translation System
        self.translator = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        self.trans_tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        self.lang_map = {
            "English": "en_XX",
            "Bengali": "bn_IN",
            "Hindi": "hi_IN",
            "Chinese": "zh_CN",
            "Spanish": "es_XX",
            "French": "fr_XX",
            "Arabic": "ar_AR",
            "Russian": "ru_RU",
        }
        self.question_generator = pipeline(
            "text2text-generation",
            model="mrm8488/t5-base-finetuned-question-generation-ap",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float32
        )

    def process_transcript(self, url, transcript):
        video_id = url.split("v=")[-1]
        if not transcript.strip():
            raise ValueError("Empty transcript received")
        
        chunks = self.chunk_text(transcript)
        if not chunks:
            raise ValueError("Failed to create text chunks")
        
        # Validate chunk content
        chunks = [c.strip() for c in chunks if c.strip()]
        
        try:
            embeddings = self.embedder.encode(chunks)
            if len(embeddings) != len(chunks):
                raise ValueError("Embedding count mismatch")
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {str(e)}")
        
        self.cache[video_id] = {
            "chunks": chunks,
            "embeddings": embeddings
        }
        if len(embeddings) != len(chunks):
            del self.cache[video_id]  # Clear invalid entry
            raise ValueError("Embedding count mismatch")


    def chunk_text(self, text, chunk_size=300):
        """Class-internal chunking with overlap"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += 1
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-int(chunk_size * 0.2):]
                current_length = len(current_chunk)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def answer_question(self, url, question):
        try:
            video_id = url.split("v=")[-1]
            if video_id not in self.cache:
                raise ValueError("❌ Please process the video transcript first")

            data = self.cache[video_id]
            
            # URGENT FIX: Proper NumPy array check
            if (not data.get("chunks") or 
                data.get("embeddings") is None or 
                data["embeddings"].size == 0):  # Critical .size check
                del self.cache[video_id]  # Force reprocessing next time
                return ("❌ Invalid context data - please reprocess video", 0.0)

            num_chunks = len(data["chunks"])
            if num_chunks == 0 or data["embeddings"].shape[0] != num_chunks:
                return ("❌ Context embedding mismatch", 0.0)

            question_embed = self.embedder.encode([question])
            similarities = cosine_similarity(question_embed, data["embeddings"])[0]
            
            valid_k = min(5, len(similarities))
            if valid_k == 0:
                return ("❌ Not enough context to answer", 0.0)

            top_idxs = np.argsort(similarities)[-valid_k:]
            context = " ".join([data["chunks"][i] for i in top_idxs[-3:]])
            
            if len(context.split()) > 500:
                context = " ".join(context.split()[:500])

            result = self.qa_model(
                question=question,
                context=context,
                max_answer_len=150,
                handle_impossible_answer=True
            )

            if result["answer"].strip() == "" or result["score"] < 0.10:
                return (f"❌ Uncertain answer - here's relevant context:\n{context[:200]}...", 0.0)
                
            return (result["answer"], result["score"])
        
        except Exception as e:
            return (f"⚠️ Critical error: {str(e)}", 0.0)



    def generate_questions(self, text, num_questions=3):
        cleaned_text = ' '.join(text.split()[:500])
        prompt = f"generate questions: {cleaned_text}"
        results = self.question_generator(
            prompt,
            max_length=80,
            num_beams=5,
            num_return_sequences=num_questions,
            early_stopping=True
        )
        return [q['generated_text'].strip().replace('question: ', '') 
                for q in results if '?' in q['generated_text']]

    def translate_text(self, text, target_lang):
        if target_lang not in self.lang_map:
            raise ValueError(f"Unsupported language: {target_lang}")

        self.trans_tokenizer.src_lang = "en_XX"
        encoded = self.trans_tokenizer(text, return_tensors="pt")
        generated_tokens = self.translator.generate(
            **encoded,
            forced_bos_token_id=self.trans_tokenizer.lang_code_to_id[
                self.lang_map[target_lang]
            ],
        )
        return self.trans_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

def generate_mindmap(summary):
    """Global mindmap generation function"""
    words = [word for word in summary.split() if len(word) > 3][:15]
    G = nx.Graph()
    for i, word in enumerate(words):
        G.add_node(word)
        if i > 0:
            G.add_edge(words[i-1], word)
    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(
        G, pos, 
        with_labels=True, 
        node_size=2500, 
        node_color="skyblue", 
        font_size=10,
        font_weight="bold", 
        edge_color="gray"
    )
    plt.savefig("mindmap.png", bbox_inches="tight")
    plt.close()
    return "mindmap.png"

# Singleton agent instance
qa_agent = QAAgent()
