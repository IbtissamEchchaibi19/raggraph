import os

def delete_files(directory):
    """
    Recursively search through directory and subdirectories to delete
    all JSON files and files named 'raw_text_cleaned'
    """
    # Count for reporting
    deleted_json_count = 0
    deleted_raw_text_count = 0
    
    # Walk through directory structure
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if file is JSON
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_json_count += 1
                    print(f"Deleted JSON file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            
            # Check if file is named 'raw_text_cleaned.txt'
            elif file == 'raw_text_cleaned.txt':
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_raw_text_count += 1
                    print(f"Deleted raw_text_cleaned file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Deleted {deleted_json_count} JSON files")
    print(f"Deleted {deleted_raw_text_count} 'raw_text_cleaned' files")
    print(f"Total deleted files: {deleted_json_count + deleted_raw_text_count}")

if __name__ == "__main__":
    # Use the specific directory path
    NLP_DIRECTORY = "C:/Users/ibtis/OneDrive/Desktop/ProccedData/NLP"
    
    # Confirm before proceeding
    print(f"This will delete all JSON files and files named 'raw_text_cleaned' in:")
    print(f"{NLP_DIRECTORY} and all subdirectories")
    confirmation = input("Do you want to continue? (y/n): ")
    
    if confirmation.lower() == 'y':
        delete_files(NLP_DIRECTORY)
    else:
        print("Operation cancelled.")