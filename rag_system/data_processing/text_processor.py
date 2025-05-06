import os
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from database.neo4j_store import store_text_chunk, link_chunk_to_author, link_chunk_to_topic

def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if start > 0 and end < len(text):
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            else:
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2
        
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def process_text_file(file_path, embedding_model):
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
        for chunk in chunks:
            embedding = embedding_model.encode(chunk).tolist()
            chunk_id = store_text_chunk(chunk, embedding, metadata)
            chunk_ids.append(chunk_id)
        print(f"Processed {file_path}: {len(chunks)} chunks created")
        return chunk_ids
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def process_text_directory(directory_path, embedding_model=None):
    if embedding_model is None:
        embedding_model = load_embedding_model()
    
    result = {}
    
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename == 'raw_text_cleaned.txt':
                file_path = os.path.join(root, filename)
                chunk_ids = process_text_file(file_path, embedding_model)
                result[filename] = chunk_ids
    
    return result