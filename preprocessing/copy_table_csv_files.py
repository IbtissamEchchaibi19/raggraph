import os
import shutil
import re
import glob

def copy_table_csv_files(source_root, target_root):
    """
    Goes through each subfolder in source_root, finds CSV files with names like table.csv, table1.csv, 
    table_1.csv, etc., and copies them to corresponding subfolders in target_root.
    
    Example:
    source_root/interik/table.csv -> target_root/interik/table.csv
    source_root/interik/table_1.csv -> target_root/interik/table_1.csv
    """
    # Create target root directory if it doesn't exist
    if not os.path.exists(target_root):
        os.makedirs(target_root)
        print(f"Created target directory: {target_root}")
    
    # Count files copied
    files_copied = 0
    
    # Walk through all subfolders recursively
    for root, dirs, files in os.walk(source_root):
        # Get the relative path from source_root
        rel_path = os.path.relpath(root, source_root)
        if rel_path == '.':  # If we're at the source_root itself
            continue
            
        # Check all files in this subfolder
        files_found = False
        for file_name in files:
            # Check if the file starts with "table" and ends with ".csv"
            if file_name.lower().startswith('table') and file_name.lower().endswith('.csv'):
                # Create corresponding subfolder in target directory if it doesn't exist
                target_subfolder = os.path.join(target_root, rel_path)
                if not os.path.exists(target_subfolder):
                    os.makedirs(target_subfolder)
                
                # Copy the file
                source_file_path = os.path.join(root, file_name)
                target_file_path = os.path.join(target_subfolder, file_name)
                shutil.copy2(source_file_path, target_file_path)
                files_copied += 1
                print(f"Copied: {rel_path}/{file_name}")
                files_found = True
        
        if not files_found:
            print(f"No table CSV files found in {rel_path}")
    
    print(f"\nFinished! Copied {files_copied} table CSV files.")

if __name__ == "__main__":
    # Set your source and target directories here
    source_directory = "C:/Users/ibtis/OneDrive/Desktop/TestLab/extrarctedcontent"
    target_directory = "C:/Users/ibtis/OneDrive/Desktop/ProccedData"
    
    copy_table_csv_files(source_directory, target_directory)