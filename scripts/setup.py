import subprocess
from pathlib import Path

def setup_environment():
    """Install required packages"""
    requirements = [
        "mlx==0.0.8",
        "mlx-lm==0.0.3",
        "streamlit==1.24.0",
        "pandas==2.0.0",
        "numpy==1.24.0",
        "sentence-transformers==2.2.2",
        "scikit-learn==1.2.0",
        "nltk==3.8.1",
        "rouge_score==0.1.2",
        "py-rouge==1.1",
        "transformers==4.31.0",
        "torch==2.0.0",
        "tqdm==4.65.0",
        "pyyaml==6.0",
        "python-dotenv==1.0.0",
        "evaluate==0.4.0",
        "bert_score==0.3.13"

    ]
    
    subprocess.check_call(["pip", "install"] + requirements)