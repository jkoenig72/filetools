import os
import sqlite3
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from moviepy.video.io.VideoFileClip import VideoFileClip
import argparse
import time

##################################################################################################################
def create_table(conn):
    """Create the 'videos' table in the database if it does not already exist.
    
    Args:
        conn (sqlite3.Connection): SQLite connection object
    """
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS videos
                      (id INTEGER PRIMARY KEY,
                       full_path TEXT NOT NULL,
                       filename TEXT NOT NULL,
                       folder TEXT NOT NULL,
                       duration INTEGER NOT NULL,
                       filesize INTEGER,
                       resolution TEXT,
                       groupid TEXT)''')
    conn.commit()

##################################################################################################################
def add_movie_to_db(conn, full_path, filename, folder, duration, filesize, resolution):
    """Add a video to the database if it does not already exist.
    
    Args:
        conn (sqlite3.Connection): SQLite connection object
        full_path (str): Full path of the movie file
        filename (str): Filename of the movie file
        duration (int): Duration of the movie in seconds
    """
    cursor = conn.cursor()
    # If the movie is not in the database, insert a new row with its details
    cursor.execute("INSERT INTO videos (full_path, filename, folder, duration, filesize, resolution, groupid) VALUES (?, ?, ?, ?, ?, ?, ?)", (full_path, filename, folder, duration, filesize, resolution, "not calculated"))
    conn.commit()

##################################################################################################################
def analyze_movie(args):
    filename, filepath, folder, dbname, total_files, file, i = args
    
    try:
        conn = sqlite3.connect(dbname)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE full_path=?", (filepath,))
            result = cursor.fetchone()

            if result is None:
                video = VideoFileClip(filepath)
                duration = int(video.duration)
                filesize = os.path.getsize(filepath)
                resolution = video.size
                resolution_str = f"{resolution[0]}x{resolution[1]}"
                video.close()

                fn = os.path.basename(filepath)
                print(f"Looking at {filename} ...")
                add_movie_to_db(conn, filepath, fn, folder, duration, filesize, resolution_str)

                i += 1
                print(f"[{i}/{total_files}] {filename} ({duration} s, {filesize} bytes)")
                file.write(f"[{i}/{total_files}] {filename} ({duration} s, {filesize} bytes)\n")

            else:
                print(f"Skipping {filename} because it already exists in the database.")
                file.write(f"Skipping {filename} because it already exists in the database.\n")

        except UnicodeEncodeError as ue:
            print(f"Unicode error with file {filepath}: {ue}. Skipping file.")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
        finally:
            conn.close()

    except Exception as e:
        print(f"Error connecting to database for {filepath}: {e}")

##################################################################################################################
def find_same_playtime_files(directory, dbname, logfilename):
    i = 0
    with open(logfilename, 'a') as file:
        # Gather all movie files in the directory
        movie_files = [os.path.join(root, f)
                       for root, dirs, files in os.walk(directory)
                       for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv', '.m4v', '.mpeg', '.m2ts', '.mpg', '.divx'))]

        total_files = len(movie_files)

        for filename in movie_files:
            filepath = os.path.join(directory, filename)
            folder = os.path.basename(os.path.dirname(filepath))

            # Process each file sequentially
            try:
                print(f"Processing file {filename} ...")
                analyze_movie([filename, filepath, folder, dbname, total_files, file, i])
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

            i += 1  # Increment the counter after each file is processed


##################################################################################################################
def compare_multiple_frames(frames1, frames2, filename1, filename2):
    """
    Compares sets of three frames from two videos and determines if they are 'same' or 'different',
    including filenames in the comparison output.
    
    :param frames1: Tuple of three frames from the first video.
    :param frames2: Tuple of three frames from the second video.
    :param filename1: Filename of the first video.
    :param filename2: Filename of the second video.
    :return: 'same' if all corresponding frames are considered similar, 'different' otherwise.
    """
    def compare_frames(frame1, frame2):
        if frame1 is None or frame2 is None:
            return "different"
        frame1_resized = cv2.resize(frame1, (0, 0), fx=0.75, fy=0.75)
        frame2_resized = cv2.resize(frame2, (frame1_resized.shape[1], frame1_resized.shape[0]), interpolation=cv2.INTER_LINEAR)
        mse = np.mean((frame1_resized - frame2_resized) ** 2)
        hist1 = cv2.calcHist([frame1_resized], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2_resized], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_comparison = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 1000
        
        # Print detailed comparison information including filenames
        print(f"Comparing {filename1} with {filename2}: MSE - {mse}, Histogram - {hist_comparison}")
        
        #if mse <= 30 or hist_comparison > 930 or (mse > 30 and mse <= 70 and hist_comparison > 190):
        if mse <= 50:
            return "same"
        else:
            return "different"
    
    # Compare each set of corresponding frames
    results = [compare_frames(f1, f2) for f1, f2 in zip(frames1, frames2)]
    
    # Print summary of comparison results with filenames
    if results.count("same") == len(frames1):
        print(f"All corresponding frames between {filename1} and {filename2} are similar. Videos classified as 'same'.")
        return "same"
    else:
        print(f"Not all corresponding frames between {filename1} and {filename2} are similar. Videos classified as 'different'.")
        return "different"

##################################################################################################################
def group_similar_videos_by_frames(video_info_with_frames):
    groups = []  # To hold groups of similar videos
    video_list = video_info_with_frames[:]  # Copy the list to avoid modifying the original

    while video_list:
        reference_video = video_list.pop(0)  # Select and remove the first video as the reference
        current_group = [reference_video]  # Start a new group with the reference video

        # Compare the reference video with the rest of the list
        i = 0  # Use an index to iterate so we can remove items without issues
        while i < len(video_list):
            comparison_result = compare_multiple_frames(
                reference_video[3],  # Frames of the reference video
                video_list[i][3],  # Frames of the current video
                reference_video[1],  # Filename of the reference video
                video_list[i][1]  # Filename of the current video
            )

            if comparison_result == "same":
                # If similar, add to the current group and remove from the list
                current_group.append(video_list.pop(i))
            else:
                # Move to the next video if not similar
                i += 1

        # Add the formed group to the list of groups
        groups.append(current_group)

    return groups



##################################################################################################################
def remove_duplicate_paths(groups):
    unique_groups = []
    for group in groups:
        unique_group = []
        seen_paths = set()  # To track seen paths
        for video in group:
            full_path = video[0]  # Assuming the full path is the first element in the tuple
            if full_path not in seen_paths:
                unique_group.append(video)
                seen_paths.add(full_path)
            else:
                print(f"#####################################################")
                print(f"Found existing path: {full_path}")
                print(f"#####################################################")
                cv2.waitKey(0)    
        unique_groups.append(unique_group)
    return unique_groups

##################################################################################################################
def capture_frames(video_path):
    """
    Capture three frames from the video: 15 seconds in, middle, and 15 seconds before the end.
    :param video_path: Path to the video file.
    :return: A tuple of three frames (start, middle, end) or None values if capture fails.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        video = VideoFileClip(video_path)
        duration = video.duration
        
        timestamps = [
            int((duration / 4) * 1000),  # 1/4 of the duration
            int((duration / 4/3) * 1000)  # 1/2 of the duration (middle)
            #int((duration * 3 / 4) * 1000)  # 3/4 of the duration
        ]
        
        frames = []

        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts)
            success, frame = cap.read()
            if success:
                frames.append(frame)
            else:
                frames.append(None)  # Append None if frame capture failed

        cap.release()
        return tuple(frames)
    except Exception as e:
        print(f"Error capturing frames for {video_path}: {e}")
        return (None, None, None)

##################################################################################################################
def sort_videofiles(dbname, logfilename, mode, min_duration=0):
    with sqlite3.connect(dbname) as conn, open(logfilename, 'a') as logfile:
        # Other initial setup code remains the same

        cursor = conn.cursor()
        # Updated SQL query to include a WHERE clause for duration
        cursor.execute("""
            SELECT duration, GROUP_CONCAT(full_path), GROUP_CONCAT(filename), GROUP_CONCAT(filesize)
            FROM videos
            WHERE duration >= ?
            GROUP BY duration
            HAVING COUNT(*) > 1
            ORDER BY duration
        """, (min_duration,))  # Pass min_duration as a parameter to the SQL query

        for duration, full_paths_str, filenames_str, filesizes_str in cursor.fetchall():
            full_paths = full_paths_str.split(',')
            filenames = filenames_str.split(',')
            filesizes = list(map(int, filesizes_str.split(',')))

            video_info_with_frames = []

            for video_path, filename, filesize in zip(full_paths, filenames, filesizes):
                frames = capture_frames(video_path)  # Capture three frames
                frame_status = "Success" if all(frame is not None for frame in frames) else "Failed"
                
                video_info_with_frames.append((video_path, filename, filesize, frames, frame_status))

                formatted_output = f"Duration: {duration}, Name: {filename}, Path: {video_path}, Size: {filesize} bytes, Frame Capture: {frame_status}"
                print(formatted_output, file=logfile)
            
            
            # Sort video_info_with_frames by filesize in descending order
            video_info_with_frames.sort(key=lambda x: x[2], reverse=True)
    

            # Filter out videos with frame capture "Failed"
            filtered_video_info = [video for video in video_info_with_frames if video[4] == "Success"]
            
             # Proceed with the rest only if there are at least two videos with successful frame captures
            if len(filtered_video_info) > 1:
                # Print the filtered list
                print(f"Filtered Videos for duration {duration} seconds, removing failed frame captures:", file=logfile)
                for path, name, size, _, status in filtered_video_info:
                    formatted_output = f"Filtered - Duration: {duration}, Name: {name}, Path: {path}, Size: {size} bytes, Frame Capture: {status}"
                    print(formatted_output)

                # Group similar videos by comparing their frames using the filtered list
                groups = group_similar_videos_by_frames(filtered_video_info)
                
                #groups = remove_duplicate_paths(groups)

                # Process the groups based on 'view' or 'write' mode
                if mode == 'view':
                    for group in groups:
                        print(f"Group with {len(group)} videos:")
                        for _, name, _, _, _ in group:
                            print(f" - {name}")
                        view_and_decide(group)
                    #input()
                elif mode == 'write':
                    # Placeholder for database update logic or other 'write' mode actions
                    pass
            else:
                print(f"Not enough videos with successfully captured frames for duration {duration} to proceed with comparison.")

        print("Operation completed.", file=logfile)

##################################################################################################################
def view_and_decide(video_group):
    """
    Display sets of three frames (start, middle, end) for each video in the group in rows, allowing the user to decide
    which ones to keep. Deletes all but the first video unless the user chooses to keep all.
    """
    if len(video_group) <= 1:
        print("Only one video in the group, nothing to decide.")
        return  # Exit if the group doesn't have multiple videos for comparison

    try:
        all_videos_frames = []  # This will store rows of frames for all videos

        for _, _, _, frames, _ in video_group:
            if any(frame is None for frame in frames):
                print("One or more frames are missing, skipping video.")
                continue

            # Resize and prepare the frames for display
            resized_frames = [cv2.resize(frame, (640, 480)) for frame in frames if frame is not None]
            # Concatenate the three frames horizontally to make a single row for the video
            video_frames_row = np.hstack(resized_frames)
            all_videos_frames.append(video_frames_row)

        if not all_videos_frames:
            print("No valid frames to display.")
            return

        # Concatenate all videos' frames vertically to display them together
        output_image = np.vstack(all_videos_frames)

        # Display the concatenated frames
        window_name = 'Grouped Videos - Decide'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, output_image)

        print("Press 'Enter' twice to delete all files except the first one, or 'Space' to cancel. If no action is taken within 15 seconds, deletion will proceed automatically.")

        start_time = time.time()
        key_press_count = 0  # To count the number of times 'Enter' is pressed
        action_taken = False  # Flag to determine whether to proceed with deletion
        last_dot_time = time.time()  # Initialize the time we last printed a dot

        while True:
            key = cv2.waitKey(100)  # Check for key press every 100 milliseconds
            if key != -1:  # If any key is pressed
                if key == 13:  # Enter key is pressed
                    key_press_count += 1
                    if key_press_count == 2:  # If 'Enter' is pressed twice
                        print("Deletion confirmed by user. Proceeding...")
                        action_taken = True
                        break  # Proceed to delete files
                elif key == 32:  # Space key is pressed
                    print("Deletion cancelled by user. Keeping all files.")
                    return  # Exit function and keep all files

            current_time = time.time()
            if current_time - last_dot_time >= 2:  # Print a dot every 2 seconds
                print(".", end='', flush=True)
                last_dot_time = current_time  # Update the last dot time

            if current_time - start_time > 3:  # 15 seconds timer
                print("\nNo input received within 3 seconds. Proceeding with automatic deletion.")
                action_taken = True
                break  # Exit the loop to proceed with automatic deletion

        # Proceed with deletion if the action is confirmed by user or by timeout
        if action_taken:
            for video_info in video_group[1:]:  # Skip the first video
                video_path, _, _, _, _ = video_info
                if os.path.exists(video_path):
                    os.remove(video_path)
                    print(f"\nDeleted {video_path}")
            print("Deletion completed.")

        cv2.destroyAllWindows()
        
    except cv2.error as e:
        print(f"Error displaying frames: {e}")
 
##################################################################################################################
def remove_duplicate_entries(conn):
    cleanup_query = """
    DELETE FROM videos
    WHERE id NOT IN (
        SELECT MIN(id)
        FROM videos
        GROUP BY full_path
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(cleanup_query)
        conn.commit()
        print("Duplicate entries removed successfully.")
    except Exception as e:
        print(f"Error during cleanup: {e}")        
            
##################################################################################################################
def print_duplicate_entries(conn):
    # Query to find duplicate full_path values
    find_duplicates_query = """
    SELECT full_path
    FROM videos
    GROUP BY full_path
    HAVING COUNT(full_path) > 1;
    """
    try:
        cursor = conn.cursor()
        cursor.execute(find_duplicates_query)
        # Fetch all duplicate full_paths
        duplicate_paths = cursor.fetchall()

        if len(duplicate_paths) == 0:
            print("No duplicates found!!!!!!!!!!!!")
        
        # For each duplicate full_path, fetch and print all corresponding rows
        for (full_path,) in duplicate_paths:
            print(f"\nDuplicate entries for {full_path}:")
            cursor.execute("SELECT * FROM videos WHERE full_path=?", (full_path,))
            duplicates = cursor.fetchall()
            for duplicate in duplicates:
                print(duplicate)
        
                
    except Exception as e:
        print(f"Error printing duplicates: {e}")
        
##################################################################################################################
def process_videos_directory(work_dir, use_existing_db=False, sort_done_db=False):
    final_directory_name = os.path.basename(os.path.normpath(work_dir))
    dbname = final_directory_name + '_videos.db' 
    logfilename = final_directory_name + '_videos.log' 

    if not use_existing_db:
        find_same_playtime_files(work_dir, dbname, logfilename)
        print("find_same_playtime_files - Done.")
    
    if not sort_done_db:
        sort_videofiles(dbname, logfilename)
        print("sort_videofiles - Done.")

runmode = "view"

parser = argparse.ArgumentParser(description='Process database file, create mode and work mode.')

# Define the -c and -w arguments
parser.add_argument('-c', '--create-mode', type=str, metavar='PATH', help='create mode')
parser.add_argument('-d', '--duration', type=int, default=0, help='Minimum duration (in seconds) for videos to be processed. Optional.')
parser.add_argument('-w', '--work-mode', nargs=2, metavar=('MODE', 'PATH'), help='work mode')

# Define the -d argument

args = parser.parse_args()

min_duration = args.duration  # Minimum duration, defaults to 0 if not specified


if args.create_mode:
    final_directory_name = os.path.basename(os.path.normpath(args.create_mode))
    dbname = final_directory_name + '_videos.db' 
    logfilename = final_directory_name + '_videos.log' 
    # Connect to the database
    conn = sqlite3.connect(dbname)
    create_table(conn)
    conn.close()
    print("Creating database - Start.")
    find_same_playtime_files(args.create_mode, dbname, logfilename)
    print("Creating database - Done.")
    #sort_videofiles(dbname, logfilename, 'write')
    
else:
    mode, path = args.work_mode
    if mode not in ['view', 'write']:
        print("Error: Invalid mode specified for work mode.")
        exit(1)
    print(f"Working on database at {path} in {mode} mode.")
    dbname = path
    root, ext = os.path.splitext(path) 
    logfilename = root + '.log' 
    if mode =='view':
        runmode = "view"
        sort_videofiles(dbname, logfilename, runmode, min_duration)
    if mode =='write':
        runmode = "write"
        sort_videofiles(dbname, logfilename, runmode, min_duration)


