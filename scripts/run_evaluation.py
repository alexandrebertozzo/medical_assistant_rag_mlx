import sys
import logging
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.evaluation.evaluator import evaluate_responses
from src.models.rag_system import MedicalRAGSystem
from src.evaluation.retieval_metrics import evaluate_retrieval

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def prepare_datasets():
    """Prepare train and test datasets"""
    try:
        # Setup paths
        data_dir = project_root / 'data'
        raw_data_path = data_dir / 'raw' / 'intern_screening_dataset.csv'
        processed_dir = data_dir / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv(raw_data_path)
        
        # Split
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,  # 20% for testing
            random_state=42
        )
        
        # Save
        train_df.to_csv(processed_dir / 'train_set.csv', index=False)
        test_df.to_csv(processed_dir / 'test_set.csv', index=False)
        
        print(f"Dataset split complete:")
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        return test_df
        
    except Exception as e:
        print(f"Error preparing datasets: {e}")
        raise

def run_evaluation(test_data: pd.DataFrame, sample_size: int = 50):
    """Run evaluation on the QA system"""
    try:
        print(f"\nStarting evaluation with sample size {sample_size}")
        
        if sample_size and sample_size < len(test_data):
            test_data = test_data.sample(n=sample_size, random_state=42)
        
        questions = test_data['question'].tolist()
        references = test_data['answer'].tolist()
        
        # Initialize system
        print("Initializing RAG system...")
        rag_system = MedicalRAGSystem()
        
        # Generate predictions and collect full responses
        print("Generating responses...")
        predictions = []
        responses = []  # Store full response objects
        for question in tqdm(questions):
            response = rag_system.generate_response(question)
            predictions.append(response['response'])
            responses.append(response)  # Store full response object
        
        # Run evaluation
        print("Calculating metrics...")
        results = evaluate_responses(
            questions,
            predictions,
            references
        )
        
        print("\n=== Evaluation Results ===")
        print(f"Sample Size: {results.sample_size}")
        print(f"\nBLEU Score: {results.bleu:.4f} (±{results.std_dev['bleu']:.4f})")
        print(f"METEOR Score: {results.meteor:.4f} (±{results.std_dev['meteor']:.4f})")
        print("\nROUGE Scores:")
        print(f"ROUGE-1: {results.rouge1:.4f} (±{results.std_dev['rouge1']:.4f})")
        print(f"ROUGE-2: {results.rouge2:.4f} (±{results.std_dev['rouge2']:.4f})")
        print(f"ROUGE-L: {results.rougeL:.4f} (±{results.std_dev['rougeL']:.4f})")

        # Calculate retrieval metrics using all responses
        retrieval_metrics = evaluate_retrieval(responses, test_data)
    
        print("\n=== Retrieval Metrics ===")
        for k in [1, 3, 5]:
            print(f"\nAt K={k}:")
            print(f"Precision@{k}: {retrieval_metrics[f'P@{k}']:.3f}")
            print(f"MAP@{k}: {retrieval_metrics[f'MAP@{k}']:.3f}")
            print(f"NDCG@{k}: {retrieval_metrics[f'NDCG@{k}']:.3f}")
        
        results_dir = project_root / 'evaluation_reports'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results including retrieval metrics
        results_df = pd.DataFrame({
            'question': questions,
            'prediction': predictions,
            'reference': references,
            'bleu_score': [results.bleu] * len(questions),
            'meteor_score': [results.meteor] * len(questions),
            'rouge1_score': [results.rouge1] * len(questions),
            'rouge2_score': [results.rouge2] * len(questions),
            'rougeL_score': [results.rougeL] * len(questions),
            'contexts_used': [len(r.get('contexts', [])) for r in responses],
            'avg_similarity': [np.mean([c['similarity'] for c in r.get('contexts', [])]) if r.get('contexts') else 0 for r in responses]
        })
        
        results_path = results_dir / f'eval_results_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        # Sample comparisons
        print("\n=== Sample Comparisons ===")
        for i in range(min(3, len(questions))):
            print(f"\nQuestion: {questions[i]}")
            print(f"Prediction: {predictions[i]}")
            print(f"Reference: {references[i]}")
            print(f"Contexts used: {len(responses[i].get('contexts', []))}")
            if responses[i].get('contexts'):
                print(f"Average similarity: {np.mean([c['similarity'] for c in responses[i]['contexts']]):.3f}")
            print("-" * 80)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise

def main():
    """Run complete evaluation pipeline"""
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--sample-size', type=int, default= 300, # For quick testing
                           help='Number of samples to evaluate')
        args = parser.parse_args()
        
        print("Preparing datasets...")
        test_data = prepare_datasets()
        
        run_evaluation(test_data, args.sample_size)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()