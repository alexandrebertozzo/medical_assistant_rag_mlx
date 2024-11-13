from pathlib import Path
import logging
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.data.embeddings import EmbeddingGenerator
from src.models.rag_system import MedicalRAGSystem
from src.evaluation.metrics import MedicalQAMetrics

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Preprocess data
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.process_data()
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_gen = EmbeddingGenerator()
        embedding_gen.generate_embeddings()
        
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        rag_system = MedicalRAGSystem()
        
        # Run evaluation
        logger.info("Running evaluation...")
        metrics = MedicalQAMetrics()
        
        # Test the system with sample questions
        test_questions = [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "What causes asthma?"
        ]
        
        for question in test_questions:
            # Get response from RAG system
            context = rag_system.get_relevant_context(question)
            response = rag_system.generate_response(question, context)
            
            # Evaluate
            reference = cleaned_data[
                cleaned_data['clean_question'].str.contains(question, case=False)
            ]['clean_answer'].iloc[0]
            
            results = metrics.evaluate_response(
                question=question,
                response=response,
                reference=reference,
                medical_context=context
            )
            
            logger.info(f"\nResults for question: {question}")
            logger.info(f"Response: {response}")
            logger.info("Metrics:")
            for metric, value in results.to_dict().items():
                logger.info(f"{metric}: {value:.4f}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()