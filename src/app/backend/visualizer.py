from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as cl
from transformers import pipeline
import tempfile
import os

class VisualSummarizer:
    def __init__(self):
        self.kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

    def create_mindmap(self, text, max_nodes=15):
        keywords = self._extract_keywords(text)
        filtered_keys = self._filter_keywords(keywords, max_nodes)
        graph = self._build_graph(filtered_keys, text)
        return self._visualize_graph(graph)

    def _extract_keywords(self, text):
        kw_results = self.kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=20
        )
        ner_results = self.ner_pipeline(text)
        combined = {kw[0]: kw[1] for kw in kw_results}
        for entity in ner_results:
            combined[entity['word']] = entity['score']
        return combined

    def _filter_keywords(self, keywords, max_nodes):
        sorted_keys = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [k[0] for k in sorted_keys[:max_nodes]]

    def _build_graph(self, keywords, text):
        G = nx.Graph()
        text_lower = text.lower()
        for kw in keywords:
            G.add_node(kw)
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i+1:]:
                if f' {kw1.lower()} ' in text_lower and f' {kw2.lower()} ' in text_lower:
                    G.add_edge(kw1, kw2)
        return G

    def _visualize_graph(self, graph):
        plt.figure(figsize=(12, 8))
        if len(graph.nodes) > 0:
            partition = cl.best_partition(graph)
            pos = nx.spring_layout(graph, k=0.5)
            nx.draw_networkx_nodes(graph, pos, node_size=2500, cmap=plt.cm.RdYlBu, 
                                node_color=list(partition.values()))
            nx.draw_networkx_edges(graph, pos, alpha=0.5)
            nx.draw_networkx_labels(graph, pos, font_size=9, font_family='sans-serif')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, bbox_inches='tight')
        plt.close()
        return temp_file.name