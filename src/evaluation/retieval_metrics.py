from typing import List, Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalEvaluator:
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = self.encoder.encode([text1, text2])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

    def calculate_precision_at_k(
        self,
        retrieved_docs: List[Dict],
        relevant_doc: Dict,
        k: int = 3,
        similarity_threshold: float = 0.5
    ) -> float:
        """
        Calculate Precision@K using semantic similarity
        
        Args:
            retrieved_docs: List of retrieved documents with context
            relevant_doc: Reference document
            k: Number of top documents to consider
            similarity_threshold: Threshold for considering a document relevant
        """
        if not retrieved_docs:
            return 0.0
            
        # Get top k retrieved docs
        top_k = retrieved_docs[:k]
        
        # Combine question + answer for comparison
        relevant_text = f"{relevant_doc['question']} {relevant_doc['answer']}"
        
        # Count similar docs in top k
        relevant_count = 0
        for doc in top_k:
            retrieved_text = f"{doc['question']} {doc['answer']}"
            similarity = self.calculate_semantic_similarity(retrieved_text, relevant_text)
            if similarity >= similarity_threshold:
                relevant_count += 1
                
        return relevant_count / k

    def calculate_map(
        self,
        retrieved_docs: List[Dict],
        relevant_doc: Dict,
        k: int = 3,
        similarity_threshold: float = 0.5
    ) -> float:
        """Calculate MAP@K using semantic similarity"""
        if not retrieved_docs:
            return 0.0
            
        relevant_text = f"{relevant_doc['question']} {relevant_doc['answer']}"
        precisions = []
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs[:k], 1):
            retrieved_text = f"{doc['question']} {doc['answer']}"
            similarity = self.calculate_semantic_similarity(retrieved_text, relevant_text)
            
            if similarity >= similarity_threshold:
                relevant_count += 1
                precisions.append(relevant_count / i)
                
        return np.mean(precisions) if precisions else 0.0

    def calculate_ndcg(
        self,
        retrieved_docs: List[Dict],
        relevant_doc: Dict,
        k: int = 3
    ) -> float:
        """Calculate NDCG@K using graded relevance based on similarity"""
        def dcg_at_k(relevance_scores: List[float], k: int) -> float:
            """Calculate DCG@K"""
            dcg = 0
            for i, score in enumerate(relevance_scores[:k], 1):
                dcg += (2 ** score - 1) / np.log2(i + 1)
            return dcg

        if not retrieved_docs:
            return 0.0
            
        relevant_text = f"{relevant_doc['question']} {relevant_doc['answer']}"
        
        # Calculate relevance scores based on similarity
        relevance_scores = []
        for doc in retrieved_docs[:k]:
            retrieved_text = f"{doc['question']} {doc['answer']}"
            similarity = self.calculate_semantic_similarity(retrieved_text, relevant_text)
            relevance_scores.append(similarity)
            
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (using perfect similarities of 1.0)
        ideal_scores = sorted(relevance_scores + [1.0] * k, reverse=True)[:k]
        idcg = dcg_at_k(ideal_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(responses: List[Dict], test_data: pd.DataFrame, k_values: List[int] = [1, 3, 5]) -> Dict:
    """Evaluate retrieval performance using semantic similarity"""
    evaluator = RetrievalEvaluator()
    metrics = {}
    
    for k in k_values:
        p_at_k = []
        map_at_k = []
        ndcg_at_k = []
        
        for response, row in zip(responses, test_data.itertuples()):
            contexts = response.get('contexts', [])
            relevant_doc = {'question': row.question, 'answer': row.answer}
            
            p_at_k.append(
                evaluator.calculate_precision_at_k(contexts, relevant_doc, k)
            )
            map_at_k.append(
                evaluator.calculate_map(contexts, relevant_doc, k)
            )
            ndcg_at_k.append(
                evaluator.calculate_ndcg(contexts, relevant_doc, k)
            )
            
        metrics.update({
            f'P@{k}': np.mean(p_at_k),
            f'MAP@{k}': np.mean(map_at_k),
            f'NDCG@{k}': np.mean(ndcg_at_k)
        })
        
    return metrics