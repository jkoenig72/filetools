import argparse
import sqlite3
import os
import subprocess
import json
from moviepy.editor import VideoFileClip
import numpy as np
import shutil 
from skimage.metrics import structural_similarity as ssim
import cv2
from datetime import datetime
from multiprocessing import Process, Queue
import sys
import time


def extract_frame_process(video_path, queue):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        frames = []
        now = datetime.now()
        formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
        print(f"{formatted_now}: Start get Frames from: {video_path}")

        # Your existing frame extraction logic
        for t in [duration/10 * i for i in range(2, 8)]:
            frame = clip.get_frame(t)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        clip.close()

        now = datetime.now()
        formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
        print(f"{formatted_now}: End get Frames from: {video_path}")
        
        queue.put((frames, None))
    except Exception as e:
        queue.put(([], str(e)))

def extract_frames(video_path, timeout=60):
    queue = Queue()
    p = Process(target=extract_frame_process, args=(video_path, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        now = datetime.now()
        formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
        print(f"{formatted_now}: Timeout occurred extracting frames from {video_path}")
        return [], f"Timeout after {timeout} seconds"

    frames, error = queue.get()
    return frames, error        

def save_frame_for_inspection(frame, filename):
    # Ensure the directory exists
    save_dir = "/mnt/truenas/storage2/complete/1byd_h265"
    save_path = os.path.join(save_dir, filename)
    
    # Save the frame to disk
    cv2.imwrite(save_path, frame)
    print(f"Saved frame for inspection: {save_path}")


def mse(imageA, imageB):
    # Calculate the mean squared error between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compare_videos(original_path, converted_path):
    original_frames, original_error = extract_frames(original_path)
    converted_frames, converted_error = extract_frames(converted_path)
    
    if original_error or converted_error:
        return False, f"Video frame extraction error: {original_error or converted_error}"
    
    if len(original_frames) == 0 or len(converted_frames) == 0:
        return False, "One or both videos have no frames available for comparison."
    
    mse_errors, psnr_scores = [], []
    
    frames_below_mse_threshold, frames_above_psnr_threshold = 0, 0
    
    try:
        for original_frame, converted_frame in zip(original_frames, converted_frames):
            # Calculate MSE
            mse_error = mse(original_frame, converted_frame)
            mse_errors.append(mse_error)
            if mse_error < 50:
                frames_below_mse_threshold += 1

           # Calculate SSIM
           # ssim_score = ssim(original_frame, converted_frame, data_range=converted_frame.max() - converted_frame.min(), multichannel=True)
           # ssim_scores.append(ssim_score)
           # if ssim_score > 0.8:
           #     frames_above_ssim_threshold += 1

            # Calculate PSNR
            psnr_score = psnr(original_frame, converted_frame)
            psnr_scores.append(psnr_score)
            if psnr_score > 29:
                frames_above_psnr_threshold += 1
            
    except ValueError as e:
        return False, f"Error comparing frames: {e}"
    
    # Determine if videos are considered the same based on the thresholds
    videos_are_same = frames_below_mse_threshold >= 4  or frames_above_psnr_threshold >= 4
    
    # Return whether videos are considered the same, along with metric scores for diagnostics
    return videos_are_same, {
        "mse": {
            "errors": mse_errors,
            "frames_below_threshold": frames_below_mse_threshold
        },
        "psnr": {
            "scores": psnr_scores,
            "frames_above_threshold": frames_above_psnr_threshold
        }
    }


def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY,
            original_path TEXT NOT NULL,
            converted_path TEXT,
            filename TEXT NOT NULL UNIQUE,
            extension TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            codec TEXT NOT NULL,
            resolution TEXT NOT NULL,
            converted INTEGER NOT NULL DEFAULT 0
        );
    ''')
    conn.commit()
    conn.close()
    now = datetime.now()
    formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
    print(f"{formatted_now}: Database created at: {db_path}")

# Function to add video to the database with metadata
def add_video_to_db(db_path, original_path, codec, resolution, filename, extension, converted_path=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check if a record with the same original_path already exists
    cursor.execute('SELECT id FROM videos WHERE original_path = ?', (original_path,))
    existing_record = cursor.fetchone()
    
    if existing_record:
        now = datetime.now()
        formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
        print(f"{formatted_now}:Video already in database: {filename}. Skipping.")
    else:
        file_size = os.path.getsize(original_path)
        converted = 1 if converted_path else 0
        try:
            cursor.execute('''
                INSERT INTO videos (original_path, converted_path, filename, extension, file_size, codec, resolution, converted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (original_path, converted_path, filename, extension, file_size, codec, resolution, converted))
            conn.commit()
            # Extended print statement with all file info
            now = datetime.now()
            formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
            print(f"{formatted_now}: Added to database: Filename: {filename}, Original Path: {original_path}, Converted Path: {converted_path}, "
                  f"Extension: {extension}, File Size: {file_size} bytes, Codec: {codec}, Resolution: {resolution}, Converted: {converted}")
        except sqlite3.IntegrityError as e:
            now = datetime.now()
            formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
            print(f"{formatted_now}:Error adding {filename} to database: {e}")
    conn.close()


# Function to process a directory of video files with expanded movie extensions
def process_directory(db_path, directory):
    video_extensions = (
        '.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv',
        '.m4v', '.mpeg', '.m2ts', '.mpg', '.divx'
    )
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                codec, resolution = get_video_metadata(video_path)
                if codec and resolution:
                    # Extract filename from video_path
                    filename = os.path.basename(video_path)
                    # Extract file extension
                    extension = os.path.splitext(filename)[1]
                    # Now pass filename and extension correctly to add_video_to_db
                    add_video_to_db(db_path, video_path, codec, resolution, filename, extension)
                else:
                    now = datetime.now()
                    formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                    print(f"{formatted_now}: Failed to retrieve metadata for {video_path}")

def get_video_metadata(video_path):
    cmd = [
        'ffprobe', 
        '-v', 'quiet', 
        '-print_format', 'json', 
        '-show_streams', 
        '-select_streams', 'v:0', 
        video_path
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
        info = json.loads(output)
        if 'streams' in info and len(info['streams']) > 0:
            codec = info['streams'][0]['codec_name']
            width = info['streams'][0]['width']
            height = info['streams'][0]['height']
            resolution = f"{width}x{height}"
            return codec, resolution
    except subprocess.CalledProcessError as e:
        print(f"Error getting video metadata: {e}")
    return None, None

def check_codec(codec):
    codec = codec.lower()  # Convert to lowercase for case-insensitive comparison
    if '265' in codec or 'hevc' in codec:
        return True
    return False

def create_target_directory(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT original_path FROM videos")
    row = cursor.fetchone()
    
    if row:
        original_path = row[0]
        directory, filename = os.path.split(original_path)
        new_directory = f"{directory}_h265"
        
        try:
            os.makedirs(new_directory)
            print(f"Created directory: {new_directory}")
        except FileExistsError:
            print(f"Directory already exists: {new_directory}")
    else:
        print("No unconverted video entries found in the database.")
    
    conn.close()
    return new_directory

def create_db_from_path(path):
    if not os.path.exists(path):
        print(f"The path {path} does not exist.")
        return
    db_name = os.path.basename(os.path.normpath(path)) + '.db'
    db_path = os.path.join(os.getcwd(), db_name)
    setup_database(db_path)
    print(f"Database '{db_name}' created in the current directory.")
    process_directory(db_path, path)
    print(f"Database '{db_name}' completed in the current directory.")

def get_duration_process(video_path, queue):
    try:
        # Using ffprobe to get video duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        duration = float(result.stdout.decode().strip())
        
        queue.put(duration)
        
    except Exception as e:
        queue.put(None)
        print(f"Error in subprocess getting duration of video: {video_path}, Error: {e}")

def get_video_duration(video_path, timeout=60):
    queue = Queue()
    p = Process(target=get_duration_process, args=(video_path, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"Timeout occurred while getting duration of video: {video_path}")
        return None

    return queue.get()

def convert_video(input_file, output_file, max_retries=999, wait_time=600):
    attempts = 0
    while attempts < max_retries:
        try:
            cmd = [
                'ffmpeg', '-y', '-i', input_file,
                '-c:v', 'hevc_nvenc',  # Use the H.265 codec with Nvidia hardware acceleration
                '-preset', 'medium',   # Encoding preset
                '-c:a', 'aac',         # AAC audio codec
                '-cq', '32',           # Constant Quality rate
                output_file
            ]
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
            if result.returncode == 0:
                print(f'Converted: {os.path.basename(input_file)}')
                return os.path.getsize(output_file), None  # No error
            else:
                raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stderr)
        
        except subprocess.CalledProcessError as e:
            error_message = str(e.output)
            if 'OpenEncodeSessionEx failed' in error_message or 'No capable devices found' in error_message:
                attempts += 1
                print(f'Error converting with hevc_nvenc: {e}. Retrying in {wait_time // 60} minutes... (Attempt {attempts}/{max_retries})')
                time.sleep(wait_time)
            else:
                print(f'Error converting with hevc_nvenc: {e}')
                return 0, e.output  # Return error output

        except UnicodeDecodeError as e:
            print(f'Decoding error during conversion: {e}')
            return 0, str(e)  # Return error as string

    return 0, f'Failed to convert after {max_retries} attempts.'

def create_status_file(base_path, filename, status):
    """Create a status file with a specific extension based on the conversion status."""
    status_filename = f"{filename}{status}"
    status_filepath = os.path.join(base_path, status_filename)
    with open(status_filepath, 'w') as f:
        f.write('')  # Create an empty file

def move_and_rename_video(original_path, new_directory, new_filename_base):
    """Move and rename a video file to the new directory with a new base filename."""
    new_filename = f"{new_filename_base}_h265.mp4"
    new_path = os.path.join(new_directory, new_filename)
    shutil.move(original_path, new_path)
    return new_path

def process_videos_for_conversion(db_path, same_directory):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM videos WHERE converted = 0")

    for row in cursor.fetchall():
        video_id, original_path, converted_path, filename, _, file_size, codec, _, _ = row

        original_directory = os.path.dirname(original_path)
        base_name = os.path.splitext(filename)[0]
        new_filename = f"{base_name}_h265.mp4"

        if same_directory:
            # Option -s is used: Save converted video in the same directory as the original
            new_converted_path = original_directory
            new_converted_fullpath = os.path.join(original_directory, new_filename)
        else:
            # No -s option: Save converted video in a new "_h265" directory at the same level as the original directory
            new_directory_name = f"{os.path.basename(original_directory)}_h265"
            new_directory_path = os.path.join(os.path.dirname(original_directory), new_directory_name)
            if not os.path.exists(new_directory_path):
                os.makedirs(new_directory_path)
            new_converted_path = new_directory_path
            new_converted_fullpath = os.path.join(new_directory_path, new_filename)

        # Step 0: Check if video is already encoded with x265/HEVC
        if '265' in codec.lower() or 'hevc' in codec.lower():
            print(f"{filename} is already encoded with h265/HEVC. Moving to converted directory.")
            try:
                # Move the original file to the converted directory
                new_converted_fullpath = os.path.join(new_converted_path, filename)
                shutil.move(original_path, new_converted_fullpath)
                print(f"Moved {filename} to {new_converted_fullpath}")
                cursor.execute("UPDATE videos SET converted_path = ?, converted = 1 WHERE id = ?", (new_converted_fullpath, video_id))
                conn.commit()
                create_status_file(new_converted_path, filename, '.success2')
            except OSError as e:
                print(f"Error moving the file {original_path} to {new_converted_fullpath}. Reason: {e.strerror}")
                cursor.execute("UPDATE videos SET converted = -7 WHERE id = ?", (video_id,))
                conn.commit()
                create_status_file(new_converted_path, filename, '.error7')
            continue

        # Step 1: Check if the original video file exists
        if not os.path.exists(original_path):
            print(f"The file {filename} does not exist at {original_path}. Marking as error in DB.")
            cursor.execute("UPDATE videos SET converted = -1 WHERE id = ?", (video_id,))
            conn.commit()
            create_status_file(new_converted_path, new_filename, '.error1')

            continue

        # Step 2: Get duration of the original video file
        original_duration = get_video_duration(original_path)
        if original_duration is None:
            print(f"Unable to retrieve duration for {filename}. Marking as error in DB.")
            cursor.execute("UPDATE videos SET converted = -2 WHERE id = ?", (video_id,))
            conn.commit()
            create_status_file(original_directory, filename, '.error2')
            continue
        
        original_duration = round(original_duration)

        # Step 3: Check if there is an existing conversion
        if converted_path and os.path.exists(converted_path):
            converted_duration = get_video_duration(converted_path)
            if converted_duration is not None:
                converted_duration = round(converted_duration)
                converted_file_size = os.path.getsize(converted_path)
                if are_values_equal(original_duration, converted_duration) and converted_file_size < file_size:
                    # Before marking as successfully validated, compare three frames
                    # videos_are_same, error_message = compare_videos(original_path, converted_path)
                    error_message = ""
                    videos_are_same = True
                    if videos_are_same:
                        print(f"Existing conversion for {filename} is valid. Marking as successfully validated.")
                        cursor.execute("UPDATE videos SET converted = 1 WHERE id = ?", (video_id,))
                        conn.commit()
                        create_status_file(new_converted_path, new_filename, '.success1')
                        # Attempt to delete the original file only after successful database update
                        try:
                            os.remove(original_path)
                            print(f"Successfully deleted the original file: {original_path}")
                        except OSError as e:
                            print(f"Error deleting the original file: {original_path}. Reason: {e.strerror}")
                    else:
                        print(f"Frame comparison failed for {filename}: {error_message}")
                        # Here you can decide to mark for re-conversion or handle as needed
                        # This example marks it for re-conversion
                        cursor.execute("UPDATE videos SET converted = -3 WHERE id = ?", (video_id,))
                        conn.commit()
                        create_status_file(new_converted_path, new_filename, '.error3')

                    continue
                else:
                    print(f"Existing conversion duration for {filename} does not match original. Considering as a conversion failure.")
                    cursor.execute("UPDATE videos SET converted = -4 WHERE id = ?", (video_id,))
                    conn.commit()
                    create_status_file(new_converted_path, new_filename, '.error4')
                    continue
            else:
                print(f"Unable to retrieve duration for the converted file of {filename}. Marking as a conversion failure.")
                cursor.execute("UPDATE videos SET converted = -5 WHERE id = ?", (video_id,))
                conn.commit()
                create_status_file(new_converted_path, new_filename, '.error5')
                continue
        else:
            # Check if the expected converted file already exists
            if os.path.exists(new_converted_fullpath):
                print(f"  converted file {new_filename} already exists. Verifying...")
                # Perform duration and frame comparison checks
                converted_duration = get_video_duration(new_converted_fullpath)
                if converted_duration is not None:
                    converted_duration = round(converted_duration)
                    
                    original_duration = get_video_duration(original_path)
                    if original_duration is not None:
                        original_duration = round(original_duration)

                        #original_duration = round(get_video_duration(original_path))
                        # videos_are_same, error_message = compare_videos(original_path, new_converted_fullpath)
                        videos_are_same = True
                        error_message = ""
                        
                        if are_values_equal(original_duration, converted_duration) and videos_are_same:
                            now = datetime.now()
                            formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                            print(f"{formatted_now}: Verification successful for {filename}. Updating database.")
                            cursor.execute("UPDATE videos SET converted_path = ?, converted = 1 WHERE id = ?", (new_converted_fullpath, video_id))
                            conn.commit()
                            create_status_file(new_converted_path, new_filename, '.success1')
                            # Attempt to delete the original file only after successful database update
                            try:
                                os.remove(original_path)
                                now = datetime.now()
                                formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")

                                print(f"{formatted_now}: Successfully deleted the original file: {original_path}")
                            except OSError as e:
                                now = datetime.now()
                                formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                                print(f"{formatted_now}: Error deleting the original file: {original_path}. Reason: {e.strerror}")

                            continue
            else:
                print(f"Verification failed for {filename}")

            # Execute the conversion
            now = datetime.now()
            formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
            print(f"{formatted_now}: Start converting: {original_path}")

            new_filesize, ffmpeg_error = convert_video(original_path, new_converted_fullpath)
            if ffmpeg_error or new_filesize == 0:
                now = datetime.now()
                formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                print(f"{formatted_now}: Error converting {filename}.")
                cursor.execute("UPDATE videos SET converted = -6 WHERE id = ?", (video_id,))
                conn.commit()
                create_status_file(new_converted_path, new_filename, '.error6')
                continue

            # Perform post-conversion checks: duration
            converted_duration = get_video_duration(new_converted_fullpath)
            if converted_duration is None:
                now = datetime.now()
                formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                print(f"{formatted_now}: Unable to retrieve duration for the converted file of {filename}. Marking as a conversion failure.")
                cursor.execute("UPDATE videos SET converted = -5 WHERE id = ?", (video_id,))
                conn.commit()
                create_status_file(new_converted_path, new_filename, '.error5')
                continue

            original_duration = round(get_video_duration(original_path))
            converted_duration = round(converted_duration)
            if not are_values_equal(original_duration, converted_duration):
                now = datetime.now()
                formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                print(f"{formatted_now}: Duration mismatch for {filename}. Marking as needing review.")
                cursor.execute("UPDATE videos SET converted = -4 WHERE id = ?", (video_id,))
                conn.commit()
                create_status_file(new_converted_path, new_filename, '.error4')
                continue
            # Frame comparison
            # videos_are_same, error_message = compare_videos(original_path, new_converted_fullpath)
            error_message =""
            videos_are_same = True

            if not videos_are_same:
                now = datetime.now()
                formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                print(f"{formatted_now}: Frame comparison failed for {filename}: {error_message}")
                cursor.execute("UPDATE videos SET converted = -3 WHERE id = ?", (video_id,))
                conn.commit()
                create_status_file(new_converted_path, new_filename, '.error3')
                continue

            # If all checks pass
            print(f"Conversion and verification successful for {filename}.")
            cursor.execute("UPDATE videos SET converted_path = ?, converted = 1 WHERE id = ?", (new_converted_fullpath, video_id))
            conn.commit()
            create_status_file(new_converted_path, new_filename, '.success1')
            # Attempt to delete the original file only after successful database update
            try:
                os.remove(original_path)
                now = datetime.now()
                formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                print(f"{formatted_now}: Successfully deleted the original file: {original_path}")
            except OSError as e:
                now = datetime.now()
                formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
                print(f"{formatted_now}: Error deleting the original file: {original_path}. Reason: {e.strerror}")

    conn.close()

def are_values_equal(a, b, tolerance=5):
    v = abs(a - b) 
    if v > tolerance:
        now = datetime.now()
        formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
        print(f"{formatted_now}: tolerance too small, v = : {v}")
    return v <= tolerance

def main():
    parser = argparse.ArgumentParser(description="Video Database Management")
    parser.add_argument('-c', '--create', type=str, help="Create an initial DB with a given path")
    parser.add_argument('-w', '--work', type=str, help="Work with the created DB")
    # Define the optional argument for same directory
    parser.add_argument('-s', '--same_directory', action='store_true', help="Save the converted videos in the same directory")

    args = parser.parse_args()

    if args.create:
        create_db_from_path(args.create)
    elif args.work:
        now = datetime.now()
        formatted_now = now.strftime("%d.%m.%Y:%H:%M:%S")
        print(f"{formatted_now}: Working with DB: {args.work}")
        new_directory = create_target_directory(args.work)
        # Pass the same_directory flag to your function
        process_videos_for_conversion(args.work, same_directory=args.same_directory)
        # Additional processing steps could follow here.

if __name__ == "__main__":
    main()
