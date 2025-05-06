import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from database.neo4j_store import (
    store_text_chunk, 
    link_chunk_to_author, 
    link_chunk_to_topic,
    store_model_evaluation
)

def load_embedding_model():
    """Load the sentence transformer model for embedding text."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if start > 0 and end < len(text):
            # Try to break at paragraph boundaries
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            else:
                # If no paragraph break, try to break at sentence boundaries
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2
        
        chunks.append(text[start:end])
        start = end - chunk_overlap
    
    return chunks

def process_text_file(file_path, embedding_model):
    """Process a single text file: chunk it, embed chunks, and store in Neo4j."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        filename = os.path.basename(file_path)
        metadata = {
            'source': filename,
            'created_at': os.path.getmtime(file_path)
        }
        
        chunks = chunk_text(text)
        chunk_ids = []
        
        print(f"Processing {filename}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            chunk_id = store_text_chunk(chunk, embedding, metadata)
            chunk_ids.append(chunk_id)
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(chunks)} chunks processed")
        
        print(f"✓ Processed {file_path}: {len(chunks)} chunks created")
        return filename, chunk_ids
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return filename, []

def process_text_directory(directory_path, embedding_model=None):
    """Process all text files in a directory."""
    if embedding_model is None:
        embedding_model = load_embedding_model()
    
    chunk_mapping = {}
    
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                file_key, chunk_ids = process_text_file(file_path, embedding_model)
                chunk_mapping[file_key] = chunk_ids
    
    print(f"✓ Processed {len(chunk_mapping)} text files")
    return chunk_mapping

def process_metadata_file(file_path, chunk_mapping=None):
    """Process a single metadata file and link chunks to metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        basename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(basename)[0] + '.txt'
        
        if chunk_mapping and filename_without_ext in chunk_mapping:
            chunk_ids = chunk_mapping[filename_without_ext]
            
            # Link chunks to author
            if 'author' in metadata:
                for chunk_id in chunk_ids:
                    link_chunk_to_author(chunk_id, metadata['author'])
                print(f"  Linked {len(chunk_ids)} chunks to author: {metadata['author']}")
            
            # Link chunks to topics
            if 'topics' in metadata:
                for topic in metadata['topics']:
                    for chunk_id in chunk_ids:
                        link_chunk_to_topic(chunk_id, topic)
                print(f"  Linked {len(chunk_ids)} chunks to {len(metadata['topics'])} topics")
        
        print(f"✓ Processed metadata from {file_path}")
        return True
    except Exception as e:
        print(f"✗ Error processing metadata file {file_path}: {e}")
        return False

def process_metadata_directory(directory_path, chunk_mapping=None):
    """Process all metadata files in a directory."""
    count = 0
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                if process_metadata_file(file_path, chunk_mapping):
                    count += 1
    
    print(f"✓ Processed {count} metadata files")
    return count

def process_csv_file(file_path):
    """Process a single CSV file containing model evaluations."""
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
            
            print(f"✓ Processed {count} model evaluations from {file_path}")
            return count
        else:
            print(f"⚠ Warning: No specialized processing for CSV schema in {file_path}")
            return 0
    except Exception as e:
        print(f"✗ Error processing CSV file {file_path}: {e}")
        return 0

def process_csv_directory(directory_path):
    """Process all CSV files in a directory."""
    total_count = 0
    
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                count = process_csv_file(file_path)
                total_count += count
    
    print(f"✓ Processed {total_count} total model evaluations")
    return total_count

def ingest_all_data(text_dir=None, metadata_dir=None, csv_dir=None):
    """Run the full data ingestion pipeline."""
    embedding_model = load_embedding_model()
    chunk_mapping = {}
    
    if text_dir:
        print(f"Processing text files from {text_dir}")
        chunk_mapping = process_text_directory(text_dir, embedding_model)
    
    if metadata_dir:
        print(f"Processing metadata files from {metadata_dir}")
        process_metadata_directory(metadata_dir, chunk_mapping)
    
    if csv_dir:
        print(f"Processing CSV files from {csv_dir}")
        process_csv_directory(csv_dir)
    
    print("✓ Data ingestion complete!")