import os
import shutil

def transfer_files():
    # Define source and destination folders
    nlp_folder = "C:/Users/ibtis/OneDrive/Desktop/NLP"
    visualization_folder = "C:/Users/ibtis/OneDrive/Desktop/Visualization"
    
    # Make sure both directories exist
    if not os.path.exists(nlp_folder):
        print(f"Error: Source directory {nlp_folder} does not exist.")
        return
    
    if not os.path.exists(visualization_folder):
        print(f"Error: Destination directory {visualization_folder} does not exist.")
        return
    
    # Get list of all subdirectories in NLP folder
    nlp_subdirs = [d for d in os.listdir(nlp_folder) if os.path.isdir(os.path.join(nlp_folder, d))]
    
    # Counter for tracking operations
    files_copied = 0
    
    # Process each subdirectory
    for subdir in nlp_subdirs:
        nlp_subdir_path = os.path.join(nlp_folder, subdir)
        vis_subdir_path = os.path.join(visualization_folder, subdir)
        
        # Check if corresponding visualization subdirectory exists
        if not os.path.exists(vis_subdir_path):
            print(f"Creating matching directory: {vis_subdir_path}")
            os.makedirs(vis_subdir_path)
        
        # Find and copy the specified files
        for file in os.listdir(nlp_subdir_path):
            # Check if file is JSON or raw_text_cleaned.txt
            if file.endswith('.json') or file == 'raw_text_cleaned.txt':
                source_file = os.path.join(nlp_subdir_path, file)
                dest_file = os.path.join(vis_subdir_path, file)
                
                try:
                    # Copy the file (use shutil.copy to preserve metadata)
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied: {source_file} → {dest_file}")
                    files_copied += 1
                except Exception as e:
                    print(f"Error copying {source_file}: {e}")
    
    print(f"\nOperation complete. {files_copied} files were copied.")

if __name__ == "__main__":
    transfer_files()