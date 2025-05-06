import os
import json
from database.neo4j_store import link_chunk_to_author, link_chunk_to_topic

def process_metadata_file(file_path, chunk_mapping=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        basename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(basename)[0] + '.txt'
        if chunk_mapping and filename_without_ext in chunk_mapping:
            chunk_ids = chunk_mapping[filename_without_ext]
            if 'author' in metadata:
                for chunk_id in chunk_ids:
                    link_chunk_to_author(chunk_id, metadata['author'])
            if 'topics' in metadata:
                for topic in metadata['topics']:
                    for chunk_id in chunk_ids:
                        link_chunk_to_topic(chunk_id, topic)
        
        print(f"Processed metadata from {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing metadata file {file_path}: {e}")
        return False

def process_metadata_directory(directory_path, chunk_mapping=None):
    count = 0
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                if process_metadata_file(file_path, chunk_mapping):
                    count += 1
    
    return count