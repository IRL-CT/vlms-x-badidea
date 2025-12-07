"""
Vision Language Model Demo for Video Analysis - Resume Mode

This script processes videos from a folder and sends frames to a Vision-Language Model
(VLM) for analysis and outcome prediction. It checks an existing results CSV and only
processes frames that haven't been analyzed yet. Results are appended to the CSV file.
"""

import cv2
import time
import base64
import requests
import json
import argparse
import os
import csv
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd

class VideoAnalyzer:
    def __init__(self,
                 model="llama3.2-vision",
                 prompt="Given the scenario shown on the video, you think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)",
                 ollama_url="http://localhost:11434",
                 video_folder="./videos",
                 output_csv="reaction_results.csv",
                 frame_sample_rate=30):

        self.model = model
        self.base_prompt = prompt
        self.ollama_url = ollama_url
        self.video_folder = video_folder
        self.output_csv = output_csv
        self.frame_sample_rate = frame_sample_rate

        # Supported video formats
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

        # Results storage
        self.results = []

        # Load existing results to determine what's already been processed
        self.processed_frames = self.load_existing_results()

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_existing_results(self):
        """Load existing results CSV and return set of (video_name, frame) tuples"""
        processed = set()

        if not os.path.exists(self.output_csv):
            print(f"No existing results file found at {self.output_csv}")
            print("Will process all frames.")
            return processed

        try:
            df = pd.read_csv(self.output_csv)

            # Check if required columns exist
            if 'video_name' not in df.columns or 'frame' not in df.columns:
                print(f"Warning: CSV doesn't have expected columns. Expected 'video_name' and 'frame'.")
                print(f"Found columns: {df.columns.tolist()}")
                return processed

            # Create set of (video_name, frame) tuples
            for _, row in df.iterrows():
                processed.add((row['video_name'], int(row['frame'])))

            print(f"Loaded {len(processed)} already-processed frames from {self.output_csv}")
            print(f"Unique videos in results: {len(df['video_name'].unique())}")

        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
            print("Will process all frames.")

        return processed

    def is_frame_processed(self, video_name, frame_idx):
        """Check if a specific frame has already been processed"""
        return (video_name, frame_idx) in self.processed_frames

    def get_video_files(self):
        """Get all video files from the specified folder"""
        video_files = []
        for file_path in Path(self.video_folder).iterdir():
            if file_path.suffix.lower() in self.video_extensions:
                video_files.append(file_path)
        return sorted(video_files)

    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 for API request"""
        # Convert BGR to RGB (PIL uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Resize image to reduce payload size
        pil_img = pil_img.resize((640, 360))

        # Save image to BytesIO object
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)

        # Get base64 string
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    def analyze_frame(self, frame, video_name):
        """Send frame to Ollama for analysis"""
        try:
            # Convert frame to base64
            base64_image = self.frame_to_base64(frame)

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
            print(response.json())
            response_data = response.json()

            # Extract the response
            if 'message' in response_data:
                description = response_data['message']['content'].strip()
                return description
            else:
                return "Error: Could not process the image."

        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def process_video(self, video_path):
        """Process a single video file, skipping already-processed frames"""
        video_name = video_path.name
        print(f"\nProcessing video: {video_name}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: Could not open video {video_name}")
            return None

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")

        # Determine which frames to sample
        frame_indices = list(range(0, total_frames, self.frame_sample_rate))

        # Filter out already-processed frames
        frames_to_process = []
        frames_skipped = 0
        for frame_idx in frame_indices:
            if self.is_frame_processed(video_name, frame_idx):
                frames_skipped += 1
            else:
                frames_to_process.append(frame_idx)

        print(f"Total frames to sample: {len(frame_indices)}")
        print(f"Already processed: {frames_skipped}")
        print(f"Remaining to process: {len(frames_to_process)}")

        if not frames_to_process:
            print(f"All frames from {video_name} have already been processed. Skipping.")
            cap.release()
            return None

        # Process only the unprocessed frames
        responses = []
        video_names = []
        frame_times = []

        for i, frame_idx in enumerate(frames_to_process):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue

            print(f"Analyzing frame {i+1}/{len(frames_to_process)} (frame index: {frame_idx})...")
            response = self.analyze_frame(frame, video_name)
            responses.append(response)
            video_names.append(video_name)
            frame_times.append(frame_idx)
            print(f"Response: {response}")
            time.sleep(1)  # Small delay to avoid overwhelming the API

        cap.release()

        if not responses:
            print(f"No new frames processed for {video_name}")
            return None

        return video_names, frame_times, responses

    def save_results_to_csv(self, append=True):
        """Save results to CSV file (append mode by default)"""
        if not self.results:
            print("No new results to save.")
            return

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(self.output_csv)

        # Determine mode: append if file exists and append=True, otherwise write
        mode = 'a' if (file_exists and append) else 'w'

        # Create csv file and save results
        with open(self.output_csv, mode=mode, newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_name', 'frame', 'outcome_prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Only write header if creating new file or overwriting
            if not file_exists or not append:
                writer.writeheader()

            for result in self.results:
                writer.writerow(result)

        print(f"\nResults saved to {self.output_csv}")
        print(f"New frames processed in this run: {len(self.results)}")

    def run(self):
        """Main method to process all videos in the folder"""
        print(f"Starting video analysis with {self.model}...")
        print(f"Video folder: {self.video_folder}")
        print(f"Output CSV: {self.output_csv}")
        print(f"Prompt: {self.base_prompt}")
        print(f"Resume mode: Only processing frames not in results CSV")

        # Get all video files
        video_files = self.get_video_files()

        if not video_files:
            print(f"No video files found in {self.video_folder}")
            print(f"Supported formats: {', '.join(self.video_extensions)}")
            return

        print(f"Found {len(video_files)} video files")

        # Process each video
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'='*50}")
            print(f"Processing video {i}/{len(video_files)}")

            try:
                result = self.process_video(video_path)

                if result:
                    # Append results to the list
                    for j in range(len(result[0])):
                        self.results.append({
                            'video_name': result[0][j],
                            'frame': result[1][j],
                            'outcome_prediction': result[2][j]
                        })
                        # Add to processed set to avoid reprocessing
                        self.processed_frames.add((result[0][j], result[1][j]))

                    print(f"Processed {len(result[0])} new frames for {video_path.name}")

                    # Save results to CSV after each video (append mode)
                    self.save_results_to_csv(append=True)
                    # Clear results buffer after saving
                    self.results = []
                else:
                    print(f"No new frames to process for {video_path.name}")

            except Exception as e:
                print(f"Error processing {video_path.name}: {str(e)}")

        print(f"\n{'='*50}")
        print("Analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Video Analysis using Vision-Language Models (Resume Mode)')
    parser.add_argument('--model', type=str, default='llama3.2-vision',
                        help='Ollama model to use (default: llama3.2-vision)')
    parser.add_argument('--prompt', type=str,
                        default='Given the scenario shown on the video, you think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)',
                        help='Prompt for the vision model')
    parser.add_argument('--video-folder', type=str, default='./videos',
                        help='Folder containing video files (default: ./videos)')
    parser.add_argument('--output-csv', type=str, default='results.csv',
                        help='Output CSV file name (default: results.csv)')
    parser.add_argument('--frame-sample-rate', type=int, default=30,
                        help='Sample every Nth frame from video (default: 30)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='Ollama API URL (default: http://localhost:11434)')

    args = parser.parse_args()

    # Create video folder if it doesn't exist
    os.makedirs(args.video_folder, exist_ok=True)

    analyzer = VideoAnalyzer(
        model=args.model,
        prompt=args.prompt,
        ollama_url=args.ollama_url,
        video_folder=args.video_folder,
        output_csv=args.output_csv,
        frame_sample_rate=args.frame_sample_rate
    )

    analyzer.run()

if __name__ == "__main__":
    main()



#nohup python vlm_reactions_resume.py  --model gemma3 --video-folder '../../../data/final_cut_videos/' --output-csv './results/results_gemma3.csv' --frame-sample-rate 3 > ./logs/vlm_output_gemma3.log 2>&1 &
#nohup python vlm_reactions_resume.py  --model  deepseek-ocr:3b --video-folder '../../../data/final_cut_videos/' --output-csv './results/results_deepseek_ocr.csv' --frame-sample-rate 3 > ./logs/vlm_output_deepseek_ocr.log 2>&1 &
#nohup p