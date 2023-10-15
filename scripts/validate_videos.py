import os
import subprocess
import tqdm

def is_valid_mp4(file_path):
    """
    Uses ffmpeg to probe the file to check if it's a valid .mp4
    """
    try:
        # Using ffprobe (part of the ffmpeg suite) to check the file
        result = subprocess.run(
            ["ffprobe", "-hide_banner", "-loglevel", "error", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        if result.returncode == 0:
            return True
        return False
    except subprocess.CalledProcessError:
        return False

def validate_videos(directory):
    """
    Traverses the directory and checks each .mp4 file
    """
    mp4_files = os.listdir(directory)
    for file in tqdm.tqdm(mp4_files):
        if file.endswith('.mp4'):
            full_path = os.path.join(directory, file)
            if not is_valid_mp4(full_path):
                print(f"[INVALID] {full_path}")
                os.remove(full_path)

if __name__ == "__main__":
    directory = "/home/data/datasets/moma/videos/boxes"
    validate_videos(directory)
