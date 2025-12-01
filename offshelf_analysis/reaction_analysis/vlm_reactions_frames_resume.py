"""
Vision Language Model Demo for Frame-based Analysis - Resume Version

This script extends vlm_reactions_frames.py to support resuming from a previous run.
It checks the log file to identify which participants have already been processed
and only processes the remaining participants.

This is useful when a long-running process is interrupted or needs to be continued.
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

class FrameAnalyzerResume:
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

        # Track processed participants
        self.processed_participants = set()

    def parse_log_file(self):
        """
        Parse the log file to find which participants have already been processed.
        Returns a set of participant IDs that have been processed.
        """
        processed = set()

        if not self.log_file or not os.path.exists(self.log_file):
            print(f"Log file not found or not specified: {self.log_file}")
            print("Will process all participants.")
            return processed

        print(f"Parsing log file: {self.log_file}")

        # Pattern to match: "Processing participant: <participant_id>"
        pattern = r"Processing participant:\s+(\S+)"

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        participant_id = match.group(1)
                        processed.add(participant_id)
        except Exception as e:
            print(f"Error reading log file: {e}")
            return processed

        print(f"Found {len(processed)} already processed participants in log file")
        if processed:
            print(f"Processed participants: {sorted(processed)}")

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

    def filter_unprocessed_participants(self, all_participants):
        """
        Filter out participants that have already been processed.
        Returns dict of only unprocessed participants.
        """
        unprocessed = {}

        for participant_id, frame_files in all_participants.items():
            if participant_id not in self.processed_participants:
                unprocessed[participant_id] = frame_files

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
        print(f"Found {len(frame_files)} frames")

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
        """Save results to CSV file"""
        if not self.results:
            print("No results to save.")
            return

        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['participant', 'frame_file', 'frame_name', 'outcome_prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                writer.writerow(result)

        print(f"\nResults saved to {self.output_csv}")
        print(f"Total frames processed: {len(self.results)}")

    def load_existing_csv_results(self):
        """Load existing results from CSV file if it exists"""
        if os.path.exists(self.output_csv):
            print(f"Loading existing results from {self.output_csv}")
            try:
                with open(self.output_csv, mode='r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.results.append(row)
                print(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                print(f"Error loading existing CSV: {e}")
                self.results = []

    def run(self):
        """Main method to process all frames in the folder"""
        print(f"Starting frame analysis with {self.model}...")
        print(f"Frames folder: {self.frames_folder}")
        print(f"Output CSV: {self.output_csv}")
        print(f"Prompt: {self.base_prompt}")

        # Parse log file to find already processed participants
        self.processed_participants = self.parse_log_file()

        # Load existing CSV results if available
        self.load_existing_csv_results()

        # Get all frame files organized by participant
        all_frames_by_participant = self.get_frame_files()

        if not all_frames_by_participant:
            print(f"No frame files found in {self.frames_folder}")
            print(f"Supported formats: {', '.join(self.image_extensions)}")
            return

        # Filter to only unprocessed participants
        frames_by_participant = self.filter_unprocessed_participants(all_frames_by_participant)

        print(f"\nTotal participants in folder: {len(all_frames_by_participant)}")
        print(f"Already processed: {len(self.processed_participants)}")
        print(f"Remaining to process: {len(frames_by_participant)}")

        if not frames_by_participant:
            print("\nAll participants have already been processed!")
            return

        total_frames = sum(len(frames) for frames in frames_by_participant.values())
        print(f"Total frames to process: {total_frames}")

        # Process each unprocessed participant
        for i, (participant_id, frame_files) in enumerate(frames_by_participant.items(), 1):
            print(f"\n{'='*50}")
            print(f"Processing participant {i}/{len(frames_by_participant)}")

            try:
                participant_results = self.process_participant_frames(participant_id, frame_files)

                if participant_results:
                    self.results.extend(participant_results)
                    print(f"Processed {len(participant_results)} frames for participant {participant_id}")
                else:
                    print(f"Failed to process participant {participant_id}")

                # Save results to CSV after each participant
                self.save_results_to_csv()

            except Exception as e:
                print(f"Error processing participant {participant_id}: {str(e)}")

        # Final save
        self.save_results_to_csv()

        print(f"\n{'='*50}")
        print("Analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Frame Analysis using Vision-Language Models - Resume Version')
    parser.add_argument('--model', type=str, default='llama3.2-vision',
                        help='Ollama model to use (default: llama3.2-vision)')
    parser.add_argument('--prompt', type=str,
                        default='Given the human reaction shown in the image, do you think the situation observed by that human ends well or poorly? (Use only one word -- well or poorly -- to answer)',
                        help='Prompt for the vision model')
    parser.add_argument('--frames-folder', type=str, default='./frames',
                        help='Folder containing frame files organized by participant (default: ./frames)')
    parser.add_argument('--output-csv', type=str, default='frame_results.csv',
                        help='Output CSV file name (default: frame_results.csv)')
    parser.add_argument('--log-file', type=str, required=True,
                        help='Path to the log file to check for already processed participants')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='Ollama API URL (default: http://localhost:11434)')

    args = parser.parse_args()

    analyzer = FrameAnalyzerResume(
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


#nohup python vlm_reactions_frames_resume.py   --model llava --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_llava_3s.csv' --log-file './logs/vlm_output_frames_3s.log' >> ./logs/vlm_output_llava3.log 2>&1 &
#nohup python vlm_reactions_frames_resume.py   --model deepseek-ocr:3b --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_deepseek-ocr_3s.csv' --log-file './logs/vlm_output_deepseek_ocr_3s.log' >> ./logs/vlm_output_deepseek_ocr_3s.log 2>&1 &
#nohup python vlm_reactions_frames_resume.py   --model  deepseek-ocr:3b --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/results_deepseek_ocr_1s.csv' --log-file './logs/vlm_output_deepseek_ocr_1s.log' >> ./logs/vlm_output_deepseek_ocr_1s.log 2>&1 &
