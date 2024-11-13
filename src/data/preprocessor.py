import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import logging
from pathlib import Path
from src import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self._setup_nltk()
        
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(text)
        filtered = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered)
        
    def process_data(self, input_file: str = None):
        """Process the dataset"""
        try:
            # Load data
            input_path = RAW_DATA_DIR / (input_file or 'intern_screening_dataset.csv')
            df = pd.read_csv(input_path)
            logger.info(f"Loaded dataset from {input_path}")
            
            # Validate columns
            required_columns = {'question', 'answer'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")
            
            # Clean data
            logger.info("Cleaning data...")
            df['clean_question'] = df['question'].apply(self.clean_text)
            df['clean_answer'] = df['answer'].apply(self.clean_text)
            
            # Remove stopwords
            logger.info("Removing stopwords from questions...")
            df['clean_question'] = df['clean_question'].apply(self.remove_stopwords)
            
            # Filter empty rows
            df = df[
                (df['clean_question'].str.strip() != '') & 
                (df['clean_answer'].str.strip() != '')
            ]
            
            # Save processed data
            output_path = PROCESSED_DATA_DIR / 'cleaned_data.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned data to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

def main():
    """Main function to run data preprocessing"""
    try:
        preprocessor = DataPreprocessor()
        preprocessor.process_data()
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()