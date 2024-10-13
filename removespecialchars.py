import os
import re
import unicodedata
import argparse


def translate_umlauts(filename):
    """Translate German umlauts in the given filename.

    Args:
        filename (str): The original filename.

    Returns:
        str: The filename with German umlauts replaced.
    """
    # Mapping of German umlaut characters to their replacements
    translations = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
    }
    # Replace each German character with its corresponding translation
    for german_char, replacement in translations.items():
        filename = filename.replace(german_char, replacement)
    return filename

def remove_accents(input_str):
    """Remove accents from characters.

    Args:
        input_str (str): The string to process.

    Returns:
        str: The string with accents removed.
    """
    # Normalize the string to decompose special characters into base and accent
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    # Combine all non-accent characters back into a single string
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def clean_filename(filename):
    """Clean and format the filename by removing unwanted characters and normalizing it.

    Args:
        filename (str): The original filename.

    Returns:
        str: The cleaned and formatted filename.
    """
    # Normalize full Unicode characters to composed form (NFC)
    filename = unicodedata.normalize('NFC', filename)
    # Remove accents
    filename = remove_accents(filename)
    # Convert the filename to lowercase
    filename = filename.lower()
    # Translate German umlauts using predefined mappings
    filename = translate_umlauts(filename)
    # Remove any character that is not a letter, digit, space, dot, hyphen, underscore, or parenthesis
    filename = re.sub(r'[^a-z0-9\s()._-]', '', filename)
    # Replace all dots with underscores, except the last one which typically precedes the file extension
    filename = re.sub(r'\.(?=.*\.)', '_', filename)
    # Replace sequences of spaces, hyphens, or underscores with a single underscore
    filename = re.sub(r'[-\s_]+', '_', filename)
    return filename

def rename_files(directory):
    """Rename files in the given directory and its subdirectories by cleaning filenames.

    Args:
        directory (str): Path to the directory where filenames need to be cleaned.
    """
    # Traverse the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(directory):
        # Iterate through each file present
        for filename in filenames:
            # Clean the filename using the defined function
            new_name = clean_filename(filename)

            # Construct full paths for old and new filenames
            old_file_path = os.path.join(dirpath, filename)
            new_file_path = os.path.join(dirpath, new_name)

            # If the new path differs from the old one, proceed with renaming
            if old_file_path != new_file_path:
                # Check if the new name already exists in the directory
                if os.path.exists(new_file_path):
                    old_file_size = os.path.getsize(old_file_path)
                    new_file_size = os.path.getsize(new_file_path)

                    # Handle conflicts based on file size
                    if old_file_size > new_file_size:
                        print(f"Name conflict: '{new_name}' exists, but file to be renamed is larger. Deleting existing and renaming.")
                        os.remove(new_file_path)  # Delete the smaller existing file
                        os.rename(old_file_path, new_file_path)  # Rename the larger file
                    else:
                        print(f"Deleted smaller file '{filename}', kept '{new_name}'")
                        os.remove(old_file_path)  # Delete the smaller file being renamed
                else:
                    print(f"Renamed '{filename}' to '{new_name}'")
                    os.rename(old_file_path, new_file_path)  # Perform the renaming
            else:
                print(f"Skipped '{filename}', name remains the same.")  # No action needed if names match

if __name__ == "__main__":
    # Parse command-line arguments for directory input
    parser = argparse.ArgumentParser(description='Rename files in a directory.')
    parser.add_argument('directory', help='Path to the directory')
    args = parser.parse_args()

    # Call the renaming function with the provided directory path
    directory_path = args.directory
    rename_files(directory_path)