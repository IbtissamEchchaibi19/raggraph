import os

# Define folder and file structure
structure = {
    "rag_system": [
        "config.py",
        "main.py",
        "requirements.txt",
        {"data_processing": [
            "__init__.py",
            "text_processor.py",
            "csv_processor.py",
            "metadata_processor.py"
        ]},
        {"database": [
            "__init__.py",
            "neo4j_setup.py",
            "neo4j_store.py"
        ]},
        {"retrieval": [
            "__init__.py",
            "query_classifier.py",
            "vector_search.py",
            "cypher_generation.py",
            "answer_generation.py"
        ]}
    ]
}

def create_structure(base_path, tree):
    for item in tree:
        if isinstance(item, str):
            file_path = os.path.join(base_path, item)
            open(file_path, 'a').close()  # Create empty file
        elif isinstance(item, dict):
            for folder, contents in item.items():
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                create_structure(folder_path, contents)

# Start creating structure
base_dir = "rag_system"
os.makedirs(base_dir, exist_ok=True)
create_structure(base_dir, structure["rag_system"])

print("Folder structure created.")
