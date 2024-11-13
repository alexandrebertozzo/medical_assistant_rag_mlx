import sys
import logging
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.data.embeddings import EmbeddingGenerator
from src import RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data():
    """Setup and process required data"""
    try:
        # Check raw data
        dataset_path = RAW_DATA_DIR / 'intern_screening_dataset.csv'
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            logger.info("Please place the dataset in the data/raw directory.")
            return False
            
        # Clear processed directory
        if PROCESSED_DATA_DIR.exists():
            shutil.rmtree(PROCESSED_DATA_DIR)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Process data
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor()
        preprocessor.process_data()
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_gen = EmbeddingGenerator()
        embedding_gen.generate_embeddings()
        
        logger.info("Data setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during data setup: {e}")
        return False

if __name__ == "__main__":
    setup_data()