import os
import pandas as pd
from database.neo4j_store import store_model_evaluation

def process_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if all(col in df.columns for col in ['model', 'metric', 'score']):
            # This is a model evaluation CSV
            count = 0
            for _, row in df.iterrows():
                store_model_evaluation(
                    model_name=row['model'],
                    metric_name=row['metric'],
                    score=float(row['score'])
                )
                count += 1
            
            print(f"Processed {count} model evaluations from {file_path}")
            return count
        else:
            print(f"Warning: No specialized processing for CSV schema in {file_path}")
            return 0
    except Exception as e:
        print(f"Error processing CSV file {file_path}: {e}")
        return 0

def process_csv_directory(directory_path):
    total_count = 0
    
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                count = process_csv_file(file_path)
                total_count += count
    
    return total_count