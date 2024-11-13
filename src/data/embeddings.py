import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
import logging
from pathlib import Path
import torch
from tqdm import tqdm
from src import PROCESSED_DATA_DIR

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = SentenceTransformer(model_name)
        self.device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available() #For Apple Silicon
            else 'cpu'
        )
        self.model.to(self.device)
        
        # Setup NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    def split_into_chunks(self, text: str, max_chunk_size: int = 150) -> list:
        """Split text into smaller chunks"""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            if len(" ".join(current_chunk + [sentence])) <= max_chunk_size:
                current_chunk.append(sentence)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
        
    def generate_embeddings(self):
        """Generate and save embeddings"""
        try:
            # Load cleaned data
            input_path = PROCESSED_DATA_DIR / 'cleaned_data.csv'
            df = pd.read_csv(input_path)
            self.logger.info(f"Loaded cleaned data from {input_path}")
            
            # Process chunks
            chunk_texts = []
            chunk_embeddings = []
            
            # Process each text in batches
            for text in tqdm(df['clean_answer'], desc="Generating embeddings"):
                chunks = self.split_into_chunks(text)
                chunk_texts.extend(chunks)
                
                # Generate embeddings
                embeddings = self.model.encode(
                    chunks,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=self.device
                )
                
                # Move to CPU and convert to numpy
                embeddings = embeddings.cpu().numpy()
                chunk_embeddings.extend(embeddings)
            
            # Convert list to numpy array
            chunk_embeddings = np.array(chunk_embeddings)
            
            # Save embeddings and chunks
            np.save(PROCESSED_DATA_DIR / 'embeddings.npy', chunk_embeddings)
            pd.DataFrame({'chunk_text': chunk_texts}).to_csv(
                PROCESSED_DATA_DIR / 'embedding_data.csv',
                index=False
            )
            
            self.logger.info(f"Generated {len(chunk_embeddings)} embeddings")
            self.logger.info(f"Saved embeddings and chunks to {PROCESSED_DATA_DIR}")
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

def main():
    """Generate embeddings for the dataset""" # May take a while depending on your hardware
    try:
        generator = EmbeddingGenerator()
        generator.generate_embeddings()
        
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise

if __name__ == "__main__":
    main()