# Medical Question-Answering System

A medical assistant bot for question-answering system that combines Retrieval-Augmented Generation (RAG) with MLX optimization for Apple Silicon, providing accurate and context-aware medical information.

## Project Overview

This system leverages state-of-the-art natural language processing techniques to provide accurate medical information by:
- Retrieving relevant medical context from a curated knowledge base;
- Generating coherent and accurate responses using MLX-optimized language models;
- Maintaining high precision in information retrieval and response generation.

### Architecture

```mermaid
graph TD
    A[User Question] --> B[RAG System]
    B --> C[Context Retrieval]
    B --> D[Response Generation]
    
    C --> E[Medical Knowledge Base]
    E --> F[Context Embedding]
    F --> G[Semantic Search]
    
    D --> H[MLX LLM]
    G --> H
    H --> I[Generated Response]
```

## Technical Components

1. **Retrieval System**
   - Semantic search using sentence-transformers
   - Context ranking with similarity scoring
   - Optimal context window (K=3) based on evaluation metrics

2. **Generation System**
   - MLX-optimized LLM for Apple Silicon
   - Context-aware response generation
   - Medical domain-specific prompt engineering

3. **Data Processing**
   - Medical text preprocessing
   - Embedding generation and caching
   - Train/test split management

### Performance Metrics
## Generation Metrics

- **BLEU (0.2249)**: Selected for evaluating word-level precision, because for medical terminology accuracy where exact term matches matter;

- **METEOR (0.4283)**: Chosen for its ability to handle synonyms and paraphrases, providing a more semantic evaluation than BLEU;

- **ROUGE-1 (0.4678)**: Used to assess coverage of individual terms and concepts in the generated answers;

- **ROUGE-2 (0.3305)**: Evaluates the system's ability to maintain proper phrase constructions and terminology pairs;

- **ROUGE-L (0.3708)**: Measures the longest common subsequence, important for preserving explanation structure.

## Retrieval Metrics

- **Precision@K (P@3: 0.964)**: Quantifies the system's ability to retrieve relevant information, with K=3 chosen based on optimal context window size;

- **MAP (0.993)**: Evaluates the ranking quality of retrieved contexts, ensuring most relevant information appears first;

- **NDCG (0.780 at K=3)**: Assesses the graded relevance of retrieved medical contexts, considering both relevance and position.

## Implementation Statement
This system was developed using standard libraries and documentation, without assistance from third-party AI systems (OpenAI, Claude, etc.). The implementation relies on established methodologies in medical QA systems and NLP evaluation metrics.

## Dataset

### Download Instructions

1. **Dataset Files**
The processed dataset files can be downloaded from Google Drive:
[Download Medical QA Dataset](https://drive.google.com/drive/folders/1eudTvOwqBvgisOE9I1oaJivV5Gu22fqj?usp=sharing) 

The dataset includes:
- `cleaned_data.csv`
- `embedding_data.csv`
- `test_set.csv`
- `train_set.csv`
- `embeddings.npy`

2. **Setup Instructions**
```bash
# Create data directories
mkdir -p data/raw data/processed

# Download and extract the files to appropriate directories:
data/
├── raw/
│   └── intern_screening_dataset.csv
└── processed/
    ├── cleaned_data.csv
    ├── embedding_data.csv
    ├── test_set.csv
    ├── train_set.csv
    └── embeddings.npy
```

## Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd medical_qa_system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare the dataset:
```bash
python scripts/prepare_and_evaluate.py
```

## Usage

### Running the Web Interface
```bash
streamlit run src/webapp/app.py
```

### Running Evaluation
```bash
python scripts/run_evaluation.py --sample-size 300 #Or any <3282 (training test size)
```

## Key Features

1. **High Retrieval Accuracy**
   - 99.3% precision for top result
   - 96.4% precision for top-3 results
   - Consistent MAP scores across K values

2. **Strong Generation Performance**
   - METEOR score of 0.43 indicating good semantic understanding
   - ROUGE-1 score of 0.47 showing good content coverage
   - Domain-appropriate BLEU score of 0.22

3. **Optimized for Performance**
   - MLX optimization for Apple Silicon
   - Efficient context retrieval
   - Caching for improved response times

## Limitations and Future Work

1. **Areas for Improvement**
   - High standard deviations in generation metrics
   - BLEU score could be enhanced
   - Response generation time optimization

2. **Future Enhancements**
   - Implement response validation
   - Add medical entity recognition
   - Enhance prompt engineering
   - Add cross-validation for metrics
   - Implement a Reranker
   - Add a more powerful/domain-related model to use as a Judge for the answers.
  
## Acknowledgments
- Cutabazi,E.;Ni,J.;Tang, G.; Cao, W. A Review on Medical Textual Question Answering Systems Based on Deep Learning Approaches. Appl. Sci. 2021, 11, 5456. https://doi.org/10.3390/ app11125456
- Asma Ben Abacha, Pierre Zweigenbaum,MEANS: A medical question-answering system combining NLP techniques and semantic Web technologies, Information Processing & Management, Volume 51, Issue 5,2015, Pages 570-594, ISSN 0306-4573, https://doi.org/10.1016/j.ipm.2015.04.006.
- T. Dodiya and S. Jain, "Question classification for medical domain Question Answering system," 2016 IEEE International WIE Conference on Electrical and Computer Engineering (WIECON-ECE), Pune, India, 2016, pp. 204-207, doi: 10.1109/WIECON-ECE.2016.8009118.
- Jacquemart, Pierre, and Pierre Zweigenbaum. "Towards a medical question-answering system: a feasibility study." The New Navigators: From Professionals to Patients. IOS Press, 2003. 463-468.
- Mrini, Khalil, et al. "Medical Question Understanding and Answering with Knowledge Grounding and Semantic Self-Supervision." arXiv preprint arXiv:2209.15301 (2022).
- https://github.com/AhmedAbouzaid1/Medical-Question-Answering-System 
- https://huggingface.co/mlx-community 
