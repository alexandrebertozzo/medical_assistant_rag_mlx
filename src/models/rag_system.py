from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import yaml

class MedicalRAGSystem:
    """RAG Medical Assistant Bot QA System combining MLX LLM with dataset knowledge"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.load_config(config_path)
        self.initialize_components()
        self.load_dataset()
        
    def load_config(self, config_path: Optional[Path] = None):
        """Load system configuration"""
        try:
            if config_path is None:
                config_path = Path(__file__).parent.parent.parent / 'configs' / 'model_config.yaml'
                
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise
            
    def initialize_components(self):
        """Initialize MLX model and embedding model""" 
        try:
            # Initialize MLX model for generation (Apple Silicon only)
            self.model, self.tokenizer = load(self.config['model']['name'])
            
            # Initialize embedding model for retrieval
            self.embedding_model = SentenceTransformer(
                self.config['embedding_model']
            )
            
            self.logger.info("Successfully initialized all components")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
            
    def load_dataset(self):
        """Load and prepare medical dataset"""
        try:
            # Load processed dataset
            data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'cleaned_data.csv'
            self.df = pd.read_csv(data_path)
            
            # Combine questions + answers for embedding
            self.texts = [
                f"Question: {q}\nAnswer: {a}" 
                for q, a in zip(self.df['question'], self.df['answer'])
            ]
            
            # Load / generate embeddings
            embeddings_path = data_path.parent / 'embeddings.npy'
            
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
                if len(self.embeddings) != len(self.texts):
                    self.logger.info("Embeddings size don't match. Regenerating embeddings...")
                    self.generate_embeddings()
            else:
                self.generate_embeddings()
                
            self.logger.info(f"Loaded dataset with {len(self.df)} entries")
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
            
    def generate_embeddings(self):
        """Generate embeddings for the dataset"""
        try:
            self.logger.info("Generating embeddings for dataset...")
            
            # Generate embeddings
            self.embeddings = self.embedding_model.encode(
                self.texts,
                show_progress_bar=True,
                batch_size=32,
                device='mps'  # Using MPS for Apple Silicon
            )
            
            embeddings_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'embeddings.npy'
            np.save(embeddings_path, self.embeddings)
            
            self.logger.info(f"Generated and saved embeddings of shape {self.embeddings.shape}")
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
            
    def retrieve_relevant_context(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, any]]:
        """Retrieve relevant medical information from dataset"""
        try:
            # Process query
            processed_query = query.lower().strip()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                [processed_query],
                convert_to_tensor=True,
                device='mps'  # Using MPS for Apple Silicon
            ).cpu().numpy()
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k most similar entries
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Filter by minimum similarity ( can increase if needed )
            min_similarity = self.config.get('retrieval', {}).get('min_similarity', 0.3)
            
            contexts = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                if similarity >= min_similarity:
                    contexts.append({
                        'question': self.df.iloc[idx]['question'],
                        'answer': self.df.iloc[idx]['answer'],
                        'similarity': similarity
                    })
            
            return contexts
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            return []
            

    def generate_response(
        self,
        question: str,
        temperature: Optional[float] = None
    ) -> Dict[str, any]:
        """Generate medical response using RAG"""
        try:
            # Retrieve relevant context
            contexts = self.retrieve_relevant_context(question)
            
            if not contexts:
                fallback_response = {
                    'response': "I apologize, but I don't have enough specific information about this medical condition in my knowledge base.",
                    'contexts': [],
                    'confidence': 0.0
                }
                return fallback_response
                
            # Prepare prompt with context
            prompt = self.format_rag_prompt(question, contexts)
            
            # Generate response
            try:
                response = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.config['model'].get('max_tokens', 512),
                    verbose=False
                )
            except Exception as gen_error:
                self.logger.error(f"Generation error: {gen_error}")
                # Try with minimal parameters
                response = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt
                )
            ",
            cleaned_response = self.clean_response(response)
            
            # Calculate confidence based on context similarities
            confidence = np.mean([ctx['similarity'] for ctx in contexts])
            
            return {
                'response': cleaned_response,
                'contexts': contexts,
                'confidence': float(confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your question. Please try another question.",
                'error': str(e),
                'contexts': [],
                'confidence': 0.0
            }
            
            
    def format_rag_prompt(
        self,
        question: str,
        contexts: List[Dict[str, any]]
    ) -> str:
        """Format prompt with retrieved context."""
        prompt = """You are a medical assistant providing accurate, helpful information. 
        Use the following relevant medical information to answer the question.
        
        Previous relevant medical questions and answers:
        """
        
        # Retrieved contexts
        for ctx in contexts:
            prompt += f"\nQuestion: {ctx['question']}\nAnswer: {ctx['answer']}\n---"
            
        prompt += f"\nNew question: {question}\n"
        prompt += """
        Using the above information, provide a clear and comprehensive answer. 
        If the information provided isn't directly relevant to the question, 
        say that you don't have enough specific information about that condition.
        """
        
        return prompt
        
            
    def clean_response(self, response: str) -> str:
        """Clean and format the model's response"""
        try:
            # Remove any system prompts and artifacts
            response = response.replace("[/INST]", "").replace("[INST]", "")
            
            response = " ".join(response.split())
            
            if response and response[0].islower():
                response = response[0].upper() + response[1:]
            
            if any(term in response.lower() for term in ['treatment', 'medication', 'therapy']):
                disclaimer = ("\n\nPlease note: This information is for educational purposes only. "
                            "Consult healthcare professionals for medical advice.")
                response = response + disclaimer
                
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error cleaning response: {e}")
            return response.strip()

def main():
    """Test the RAG medical system"""
    try:
        # Initialize system
        system = MedicalRAGSystem()
        
        # Test questions
        test_questions = [
            "What are the common symptoms of diabetes?",
            "How is high blood pressure treated?",
            "What causes asthma attacks?"
        ]
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            result = system.generate_response(question)
            print(f"\nResponse: {result['response']}")
            print(f"\nConfidence: {result['confidence']:.2f}")
            print("\nRelevant Contexts:")
            for ctx in result['contexts']:
                print(f"- Q: {ctx['question']}")
                print(f"  A: {ctx['answer']}")
                print(f"  Similarity: {ctx['similarity']:.2f}")
            print("-" * 80)
            
    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()