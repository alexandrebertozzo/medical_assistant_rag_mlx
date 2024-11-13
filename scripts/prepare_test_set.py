import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_datasets():
    """Prepare train and test datasets."""
    try:
        # Setup paths
        data_dir = Path(__file__).parent.parent / 'data'
        raw_data_path = data_dir / 'raw' / 'intern_screening_dataset.csv'
        
        # Create processed directory
        processed_dir = data_dir / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
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
        
    except Exception as e:
        print(f"Error preparing datasets: {e}")
        raise

if __name__ == "__main__":
    prepare_datasets()