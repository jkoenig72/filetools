import os
import glob
import shutil
import sys

def is_hex_name(name):
    """Check if the name is a 32-char hexadecimal string."""
    try:
        int(name, 16)
        return len(name) == 32
    except ValueError:
        return False

def remove_brackets(name):
    """Remove square brackets from the name."""
    return name.replace("[", "").replace("]", "")

def rename_and_remove_brackets(directory):
    """Rename directories and files by removing brackets."""
    for root, dirs, files in os.walk(directory, topdown=False):
        # Rename files
        for file in files:
            new_name = remove_brackets(file)
            if new_name != file:
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)

        # Rename directories
        new_dir_name = remove_brackets(os.path.basename(root))
        if new_dir_name != os.path.basename(root):
            parent_dir = os.path.dirname(root)
            new_dir_path = os.path.join(parent_dir, new_dir_name)
            if not os.path.exists(new_dir_path):
                os.rename(root, new_dir_path)
            else:
                shutil.rmtree(root)  # Delete the directory if the new name already exists


def delete_specified_files_and_dirs(directory):
    file_extensions = [
        "*.jpeg", "*.log", "*.iso","*.bat", "*.url", "*.sfv", "*.vtx", "*.jpg", "*.nfo", "*.srr", "*.scr", "*.exe", "*.r??", "*.png", "*.par", 
        "*.nzb", "*.pdf", "*.doc", "*.zip", "*.diz", "*.html", "*.htm", "*.txt", "*.srt", 
        "*.par2", "*.url", "*.m2ts", "*.VOB", "*.0", "*.1", "*.2", "*.3", "*.4", "*.5", 
        "*.6", "*.7", "*.8", "*.9", "*.10", "*.11", "*.12", "*.psd", "*.backup", "*.bak", 
        "*.bdjo", "*.bdmv", "*.mpls", "*.BUP", "*.cci", "*.cert", "*.crl", "*.clpi", "*.conf", 
        "*.cfg", "*.db", "*.db-shm", "*.db-wal", "*.IFO", "*.vob", "*.gif", "*.hcf", "*.idx", 
        "*.jar", "*.java", "*.json", "*.md5", "*.docx", "*.miniso", "*.bin", "*.fontindex", 
        "*.lst", "*.m3u", "*.xml", "*.otf", "*.properties", "*.sig", "*.crt", "*.srs", 
        "*.sub", "*.xxx", "*.ini", "*.URL", "*.dat", "*.JPG", "*.md", "*.mp3", "*.part", 
        "*sample.*"
        # Add any other file extensions as needed
    ]

    directories_to_delete = ["_unpack", "sample", "Sample"]
    
   # Delete files with specified extensions
    for ext in file_extensions:
        for file_path in glob.glob(os.path.join(directory, '**', ext), recursive=True):
            if os.path.isfile(file_path):
                try:
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")

    # Delete specified directories, including their contents
    for dir_name in directories_to_delete:
        for dir_path in glob.glob(os.path.join(directory, '**', dir_name), recursive=True):
            if os.path.isdir(dir_path):
                print(f"Deleting directory and its contents: {dir_path}")
                shutil.rmtree(dir_path)

    # Delete empty directories
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                if not os.listdir(dir_path):
                    print(f"Deleting empty directory: {dir_path}")
                    os.rmdir(dir_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

def process_directory(directory):
    files_by_size = {}
    total_space_saved = 0

    # First pass: Process hex-named files
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            size = os.path.getsize(path)
            if (size, root) not in files_by_size:
                files_by_size[(size, root)] = []
            files_by_size[(size, root)].append(path)

    for (size, dir_path), paths in files_by_size.items():
        hex_files = [p for p in paths if is_hex_name(os.path.splitext(os.path.basename(p))[0])]
        real_files = [p for p in paths if not is_hex_name(os.path.splitext(os.path.basename(p))[0])]

        if real_files:
            for hex_file in hex_files:
                print(f"Remove duplicated Hex File: {hex_file}")
                total_space_saved += os.path.getsize(hex_file)
                os.remove(hex_file)
        elif hex_files:
            kept_file = hex_files[0]
            dir_name = os.path.basename(dir_path)
            new_name = os.path.join(dir_path, dir_name + os.path.splitext(kept_file)[1])
            print(f"Rename Hex File: {kept_file} to {new_name}")
            os.rename(kept_file, new_name)
            for hex_file in hex_files[1:]:
                print(f"Remove duplicated Hex File: {hex_file}")
                total_space_saved += os.path.getsize(hex_file)
                os.remove(hex_file)

    # Second pass: Remove files with the same size and shorter names
    for root, _, files in os.walk(directory):
        files_by_size = {}
        for file in files:
            path = os.path.join(root, file)
            size = os.path.getsize(path)
            if size not in files_by_size:
                files_by_size[size] = []
            files_by_size[size].append(path)

        for size, paths in files_by_size.items():
            if len(paths) > 1:
                paths.sort(key=lambda x: len(os.path.basename(x)))
                for file_to_remove in paths[:-1]:
                    print(f"Removing file with duplicate size: {file_to_remove}")
                    total_space_saved += os.path.getsize(file_to_remove)
                    os.remove(file_to_remove)

    # Convert bytes to gigabytes
    space_saved_gb = total_space_saved / (1024**3)
    print(f"Total space saved: {space_saved_gb:.2f} GB")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <top-level-directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"The specified directory does not exist: {directory}")
        sys.exit(1)
    
    rename_and_remove_brackets(directory)
    delete_specified_files_and_dirs(directory)
    process_directory(directory)

if __name__ == "__main__":
    main()