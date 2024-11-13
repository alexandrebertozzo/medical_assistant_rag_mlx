import numpy as np
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
import logging
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class EvaluationResults: #Run separately from the App
    bleu: float
    rouge1: float
    rouge2: float
    rougeL: float
    meteor: float
    std_dev: Dict[str, float]
    sample_size: int

class BleuEvaluator:
    """BLEU score evaluator"""
    
    def __init__(self):
        self.smoother = SmoothingFunction().method1
        
    def evaluate(
        self,
        questions: List[str],
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Calculate BLEU score"""
        scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                score = sentence_bleu(
                    [ref.split()],
                    pred.split(),
                    smoothing_function=self.smoother
                )
                scores.append(score)
            except Exception as e:
                logging.warning(f"Error calculating BLEU: {e}")
                scores.append(0.0)
                
        return {
            'score': mean(scores),
            'std_dev': stdev(scores) if len(scores) > 1 else 0
        }

class RougeEvaluator:
    """ROUGE score evaluator"""
    
    def __init__(self, rouge_types=None):
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        
    def evaluate(
        self,
        questions: List[str],
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            try:
                results = self.scorer.score(ref, pred)
                for key in scores:
                    scores[key].append(results[key].fmeasure)
            except Exception as e:
                logging.warning(f"Error calculating ROUGE: {e}")
                for key in scores:
                    scores[key].append(0.0)
                    
        return {
            key: {
                'score': mean(values),
                'std_dev': stdev(values) if len(values) > 1 else 0
            }
            for key, values in scores.items()
        }

class MeteorEvaluator:
    """METEOR score evaluator"""
    
    def __init__(self):
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
            
    def evaluate(
        self,
        questions: List[str],
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Calculate METEOR score"""
        scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                score = meteor_score([ref.split()], pred.split())
                scores.append(score)
            except Exception as e:
                logging.warning(f"Error calculating METEOR: {e}")
                scores.append(0.0)
                
        return {
            'score': mean(scores),
            'std_dev': stdev(scores) if len(scores) > 1 else 0
        }

def evaluate_responses(
    questions: List[str],
    predictions: List[str],
    references: List[str],
    sample_size: int = None
) -> EvaluationResults:
    """
    Evaluate model responses using multiple metrics
    
    Args:
        questions: List of questions
        predictions: List of model-generated answers
        references: List of ground truth answers
        sample_size: Optional size for random sampling
    """
    try:
        if sample_size:
            # Random sampling
            indices = np.random.choice(
                len(questions),
                size=min(sample_size, len(questions)),
                replace=False
            )
            questions = [questions[i] for i in indices]
            predictions = [predictions[i] for i in indices]
            references = [references[i] for i in indices]
        
        # Initialize evaluators
        bleu_eval = BleuEvaluator()
        rouge_eval = RougeEvaluator()
        meteor_eval = MeteorEvaluator()
        
        # Calculate scores
        bleu_results = bleu_eval.evaluate(questions, predictions, references)
        rouge_results = rouge_eval.evaluate(questions, predictions, references)
        meteor_results = meteor_eval.evaluate(questions, predictions, references)
        
        return EvaluationResults(
            bleu=bleu_results['score'],
            rouge1=rouge_results['rouge1']['score'],
            rouge2=rouge_results['rouge2']['score'],
            rougeL=rouge_results['rougeL']['score'],
            meteor=meteor_results['score'],
            std_dev={
                'bleu': bleu_results['std_dev'],
                'rouge1': rouge_results['rouge1']['std_dev'],
                'rouge2': rouge_results['rouge2']['std_dev'],
                'rougeL': rouge_results['rougeL']['std_dev'],
                'meteor': meteor_results['std_dev']
            },
            sample_size=len(questions)
        )
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise