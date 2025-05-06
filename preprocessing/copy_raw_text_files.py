import os
import shutil

def copy_raw_text_files(source_root, target_root):
    """
    Goes through each subfolder in source_root, finds raw_text.txt files,
    and copies them to corresponding subfolders in target_root.
    
    Example:
    source_root/interik/raw_text.txt -> target_root/interik/raw_text.txt
    """
    # Create target root directory if it doesn't exist
    if not os.path.exists(target_root):
        os.makedirs(target_root)
        print(f"Created target directory: {target_root}")
    
    # Count files copied
    files_copied = 0
    
    # Get immediate subfolders of source_root
    subfolders = [f.path for f in os.scandir(source_root) if f.is_dir()]
    
    for subfolder_path in subfolders:
        # Get the name of the subfolder
        subfolder_name = os.path.basename(subfolder_path)
        
        # Path to the raw_text.txt file
        raw_text_path = os.path.join(subfolder_path, "raw_text.txt")
        
        # Check if raw_text.txt exists in this subfolder
        if os.path.exists(raw_text_path):
            # Create corresponding subfolder in target directory
            target_subfolder = os.path.join(target_root, subfolder_name)
            if not os.path.exists(target_subfolder):
                os.makedirs(target_subfolder)
            
            # Copy the file
            target_file_path = os.path.join(target_subfolder, "raw_text.txt")
            shutil.copy2(raw_text_path, target_file_path)
            files_copied += 1
            print(f"Copied: {subfolder_name}/raw_text.txt")
        else:
            print(f"No raw_text.txt found in {subfolder_name}")
    
    print(f"\nFinished! Copied {files_copied} files.")

if __name__ == "__main__":
    # Set your source and target directories here
    source_directory = "C:/Users/ibtis/OneDrive/Desktop/TestLab/extrarctedcontent"
    target_directory = "C:/Users/ibtis/OneDrive/Desktop/ProccedData"
    
    copy_raw_text_files(source_directory, target_directory)