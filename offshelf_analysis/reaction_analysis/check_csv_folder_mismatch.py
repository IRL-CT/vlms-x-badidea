"""
Check for mismatches between CSV entries and actual frames in folder
"""
import csv
from pathlib import Path
import sys

def check_mismatch(csv_file, frames_folder):
    # Read CSV entries
    csv_frames = set()
    if Path(csv_file).exists():
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                participant = row.get('participant', '').strip()
                frame_file = row.get('frame_file', '').strip()
                if participant and frame_file:
                    csv_frames.add((participant, frame_file))

    # Read actual frames in folder
    folder_frames = set()
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    frames_path = Path(frames_folder)

    if frames_path.exists():
        for participant_dir in frames_path.iterdir():
            if participant_dir.is_dir():
                participant_id = participant_dir.name
                for frame_file in participant_dir.iterdir():
                    if frame_file.is_file() and frame_file.suffix.lower() in image_extensions:
                        folder_frames.add((participant_id, frame_file.name))

    # Find mismatches
    in_csv_not_folder = csv_frames - folder_frames
    in_folder_not_csv = folder_frames - csv_frames

    print(f"CSV file: {csv_file}")
    print(f"Frames folder: {frames_folder}")
    print(f"\nFrames in CSV: {len(csv_frames)}")
    print(f"Frames in folder: {len(folder_frames)}")
    print(f"Frames in both: {len(csv_frames & folder_frames)}")
    print(f"\nFrames in CSV but NOT in folder: {len(in_csv_not_folder)}")
    print(f"Frames in folder but NOT in CSV: {len(in_folder_not_csv)}")

    if in_csv_not_folder:
        print(f"\nFirst 10 examples of frames in CSV but not in folder:")
        for participant, frame_file in list(in_csv_not_folder)[:10]:
            print(f"  {participant}/{frame_file}")

    if in_folder_not_csv:
        print(f"\nFirst 10 examples of frames in folder but not in CSV:")
        for participant, frame_file in list(in_folder_not_csv)[:10]:
            print(f"  {participant}/{frame_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_csv_folder_mismatch.py <csv_file> <frames_folder>")
        sys.exit(1)

    check_mismatch(sys.argv[1], sys.argv[2])
