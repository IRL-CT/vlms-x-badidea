"""
Vision Language Model Demo for Frame-based Analysis - Resume Version 2

This script extends vlm_reactions_frames_resume.py to support frame-level resuming.
Unlike the original version which skips entire participants if they were started,
this version checks which specific frames are missing from the CSV and processes
only those frames, even if some frames from that video are already in the CSV.

This ensures that no frames are missed even if a previous run was interrupted
partway through processing a participant.
"""

import time
import base64
import requests
import json
import argparse
import os
import csv
import re
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path

class FrameAnalyzerResumeV2:
    def __init__(self,
                 model="llama3.2-vision",
                 prompt="Given the human reaction shown in the image, do you think the situation observed by the subject ends well or poorly? (Use only one word to answer)",
                 ollama_url="http://localhost:11434",
                 frames_folder="./frames",
                 output_csv="reaction_results.csv",
                 log_file=None):

        self.model = model
        self.base_prompt = prompt
        self.ollama_url = ollama_url
        self.frames_folder = frames_folder
        self.output_csv = output_csv
        self.log_file = log_file

        # Supported image formats
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        # Results storage
        self.results = []

        # Track processed frames at the frame level (participant, frame_file)
        self.processed_frames = set()

    def parse_csv_for_processed_frames(self):
        """
        Parse the CSV file to find which specific frames have already been processed.
        Returns a set of (participant, frame_file) tuples.
        """
        processed = set()

        if not os.path.exists(self.output_csv):
            print(f"CSV file not found: {self.output_csv}")
            print("Will process all frames.")
            return processed

        print(f"Parsing CSV file: {self.output_csv}")

        try:
            with open(self.output_csv, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    participant = row.get('participant', '').strip()
                    frame_file = row.get('frame_file', '').strip()

                    if participant and frame_file:
                        processed.add((participant, frame_file))
                        # Also load the existing results
                        self.results.append(row)

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return processed

        print(f"Found {len(processed)} already processed frames in CSV")
        print(f"Loaded {len(self.results)} existing results")

        return processed

    def get_frame_files(self):
        """
        Get all frame files organized by participant folder.
        Returns dict: {participant_id: [frame_paths]}
        """
        frames_by_participant = {}
        frames_folder_path = Path(self.frames_folder)

        # Check if folder exists
        if not frames_folder_path.exists():
            print(f"Error: Frames folder not found: {self.frames_folder}")
            return frames_by_participant

        # Iterate through participant folders
        for participant_dir in sorted(frames_folder_path.iterdir()):
            if participant_dir.is_dir():
                participant_id = participant_dir.name
                # Get all image files in this participant folder
                frame_files = []
                for frame_file in participant_dir.iterdir():
                    if frame_file.is_file() and frame_file.suffix.lower() in self.image_extensions:
                        frame_files.append(frame_file)

                if frame_files:
                    # Sort frames by name to maintain order
                    frames_by_participant[participant_id] = sorted(frame_files)

        return frames_by_participant

    def filter_unprocessed_frames(self, all_participants):
        """
        Filter out frames that have already been processed.
        Returns dict of participants with only unprocessed frames.
        """
        unprocessed = {}

        for participant_id, frame_files in all_participants.items():
            # Filter frames for this participant
            unprocessed_frames = []

            for frame_path in frame_files:
                frame_file = frame_path.name

                # Check if this specific frame has been processed
                if (participant_id, frame_file) not in self.processed_frames:
                    unprocessed_frames.append(frame_path)

            # Only add participant if they have unprocessed frames
            if unprocessed_frames:
                unprocessed[participant_id] = unprocessed_frames

        return unprocessed

    def image_to_base64(self, image_path):
        """Convert image file to base64 for API request"""
        try:
            # Check if file exists and has non-zero size
            if not image_path.exists():
                print(f"Error: Image file does not exist: {image_path}")
                return None

            if image_path.stat().st_size == 0:
                print(f"Error: Image file is empty: {image_path}")
                return None

            # Open image with error handling for truncated files
            try:
                pil_img = Image.open(image_path)
                # Force loading of the image data to catch truncated files early
                pil_img.load()
            except (OSError, IOError) as e:
                if "truncated" in str(e).lower() or "broken" in str(e).lower():
                    print(f"Warning: Truncated or corrupted image file: {image_path}")
                    try:
                        # Try to load the image with partial data
                        from PIL import ImageFile
                        ImageFile.LOAD_TRUNCATED_IMAGES = True
                        pil_img = Image.open(image_path)
                        pil_img.load()
                        print(f"Successfully loaded truncated image: {image_path}")
                    except Exception as retry_e:
                        print(f"Error: Cannot load corrupted image {image_path}: {retry_e}")
                        return None
                else:
                    raise e

            # Convert to RGB if necessary
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # Resize image to reduce payload size (optional, adjust as needed)
            pil_img = pil_img.resize((320, 180))

            # Save image to BytesIO object
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)

            # Get base64 string
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str

        except Exception as e:
            print(f"Error converting image to base64 for {image_path}: {str(e)}")
            return None

    def analyze_frame(self, frame_path):
        """Send frame to Ollama for analysis"""
        try:
            # Convert frame to base64
            base64_image = self.image_to_base64(frame_path)

            if base64_image is None:
                return "Error: Could not convert image to base64"

            # Build the prompt
            prompt = self.base_prompt

            print(f"Prompt: {prompt}")
            print(f"Image size: {len(base64_image)} bytes")

            # Prepare Ollama API request
            api_url = f"{self.ollama_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64_image]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096
                }
            }

            # Make the API request
            response = requests.post(api_url, json=payload, timeout=600)
            print(f"Response status code: {response.status_code}")
            response_data = response.json()
            print(response_data)

            # Extract the response
            if 'message' in response_data:
                description = response_data['message']['content'].strip()
                return description
            else:
                return "Error: Could not process the image."

        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def process_participant_frames(self, participant_id, frame_files):
        """Process all frames for a single participant"""
        print(f"\nProcessing participant: {participant_id}")
        print(f"Found {len(frame_files)} unprocessed frames for this participant")

        participant_results = []

        # Analyze each frame
        for i, frame_path in enumerate(frame_files):
            print(f"Analyzing frame {i+1}/{len(frame_files)}: {frame_path.name}")

            # Extract frame number from filename if possible
            # Expected format: videoname_frameXXXX.png
            frame_name = frame_path.stem

            response = self.analyze_frame(frame_path)
            print(f"Response: {response}")

            participant_results.append({
                'participant': participant_id,
                'frame_file': frame_path.name,
                'frame_name': frame_name,
                'outcome_prediction': response
            })

            time.sleep(1)  # Small delay to avoid overwhelming the API

        return participant_results

    def save_results_to_csv(self):
        """Save results to CSV file, sorted by participant and frame_file"""
        if not self.results:
            print("No results to save.")
            return

        # Sort results by participant, then by frame_file for consistent ordering
        sorted_results = sorted(self.results, key=lambda x: (x.get('participant', ''), x.get('frame_file', '')))

        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['participant', 'frame_file', 'frame_name', 'outcome_prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in sorted_results:
                writer.writerow(result)

        print(f"\nResults saved to {self.output_csv}")
        print(f"Total frames in CSV: {len(self.results)}")

    def run(self):
        """Main method to process all frames in the folder"""
        print(f"Starting frame analysis with {self.model}...")
        print(f"Frames folder: {self.frames_folder}")
        print(f"Output CSV: {self.output_csv}")
        print(f"Prompt: {self.base_prompt}")

        # Parse CSV file to find already processed frames (frame-level tracking)
        self.processed_frames = self.parse_csv_for_processed_frames()

        # Get all frame files organized by participant
        all_frames_by_participant = self.get_frame_files()

        if not all_frames_by_participant:
            print(f"No frame files found in {self.frames_folder}")
            print(f"Supported formats: {', '.join(self.image_extensions)}")
            return

        # Filter to only unprocessed frames
        frames_by_participant = self.filter_unprocessed_frames(all_frames_by_participant)

        # Calculate statistics
        total_participants_in_folder = len(all_frames_by_participant)
        total_frames_in_folder = sum(len(frames) for frames in all_frames_by_participant.values())
        participants_with_work = len(frames_by_participant)
        frames_to_process = sum(len(frames) for frames in frames_by_participant.values())

        print(f"\n{'='*60}")
        print(f"FRAME-LEVEL RESUME STATISTICS:")
        print(f"{'='*60}")
        print(f"Total participants in folder: {total_participants_in_folder}")
        print(f"Total frames in folder: {total_frames_in_folder}")
        print(f"Already processed frames: {len(self.processed_frames)}")
        print(f"Participants with unprocessed frames: {participants_with_work}")
        print(f"Frames remaining to process: {frames_to_process}")
        print(f"{'='*60}\n")

        if not frames_by_participant:
            print("All frames have already been processed!")
            return

        # Process each participant that has unprocessed frames
        for i, (participant_id, frame_files) in enumerate(frames_by_participant.items(), 1):
            print(f"\n{'='*50}")
            print(f"Processing participant {i}/{len(frames_by_participant)}: {participant_id}")

            try:
                participant_results = self.process_participant_frames(participant_id, frame_files)

                if participant_results:
                    self.results.extend(participant_results)
                    print(f"Processed {len(participant_results)} frames for participant {participant_id}")

                    # Update processed_frames set
                    for result in participant_results:
                        self.processed_frames.add((result['participant'], result['frame_file']))
                else:
                    print(f"Failed to process participant {participant_id}")

                # Save results to CSV after each participant
                self.save_results_to_csv()

            except Exception as e:
                print(f"Error processing participant {participant_id}: {str(e)}")
                # Continue with next participant even if this one fails

        # Final save
        self.save_results_to_csv()

        print(f"\n{'='*50}")
        print("Analysis complete!")
        print(f"Total frames now in CSV: {len(self.results)}")

def main():
    parser = argparse.ArgumentParser(description='Frame Analysis using Vision-Language Models - Resume Version 2 (Frame-level)')
    parser.add_argument('--model', type=str, default='llama3.2-vision',
                        help='Ollama model to use (default: llama3.2-vision)')
    parser.add_argument('--prompt', type=str,
                        default='Given the human reaction shown in the image, do you think the situation observed by that human ends well or poorly? (Use only one word -- well or poorly -- to answer)',
                        help='Prompt for the vision model')
    parser.add_argument('--frames-folder', type=str, default='./frames',
                        help='Folder containing frame files organized by participant (default: ./frames)')
    parser.add_argument('--output-csv', type=str, default='frame_results.csv',
                        help='Output CSV file name (default: frame_results.csv)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to the log file (optional, for compatibility with v1)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='Ollama API URL (default: http://localhost:11434)')

    args = parser.parse_args()

    analyzer = FrameAnalyzerResumeV2(
        model=args.model,
        prompt=args.prompt,
        ollama_url=args.ollama_url,
        frames_folder=args.frames_folder,
        output_csv=args.output_csv,
        log_file=args.log_file
    )

    analyzer.run()

if __name__ == "__main__":
    main()


# Example usage:
# nohup python vlm_reactions_frames_resume_v2.py --model llava --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_llava_3s.csv' >> ./logs/vlm_output_llava_3s_v2.log 2>&1 &
# nohup python vlm_reactions_frames_resume_v2.py --model deepseek-ocr:3b --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_deepseek-ocr_3s.csv' >> ./logs/vlm_output_deepseek_ocr_3s_v2.log 2>&1 &
# nohup python vlm_reactions_frames_resume_v2.py --model gemma3 --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_gemma3_3s.csv' >> ./logs/vlm_output_gemma3_3s_v2.log 2>&1 &
# nohup python vlm_reactions_frames_resume_v2.py --model llava-llama3 --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_llavallama3_3s.csv' >> ./logs/vlm_output_llava-llama3_3s_v2.log 2>&1 &