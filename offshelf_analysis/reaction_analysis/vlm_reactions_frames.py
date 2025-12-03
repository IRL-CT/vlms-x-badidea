"""
Vision Language Model Demo for Frame-based Analysis

This script processes pre-extracted frames from a folder and sends them to a Vision-Language Model
(VLM) for analysis and outcome prediction. The system uses Ollama to access
open-source multimodal models like Llama 3.2 Vision for completely local inference.
Results are saved to a CSV file.

Adapted from vlm_reactions.py to work with pre-extracted frames instead of videos.
"""

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

class FrameAnalyzer:
    def __init__(self,
                 model="llama3.2-vision",
                 prompt="Given the human reaction shown in the image, do you think the situation observed by the subject ends well or poorly? (Use only one word to answer)",
                 ollama_url="http://localhost:11434",
                 frames_folder="./frames",
                 output_csv="reaction_results.csv"):

        self.model = model
        self.base_prompt = prompt
        self.ollama_url = ollama_url
        self.frames_folder = frames_folder
        self.output_csv = output_csv

        # Supported image formats
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        # Results storage
        self.results = []

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
                    #print(f"Found file: {frame_file.name}")
                    if frame_file.is_file() and frame_file.suffix.lower() in self.image_extensions:
                        frame_files.append(frame_file)

                if frame_files:
                    # Sort frames by name to maintain order
                    frames_by_participant[participant_id] = sorted(frame_files)

        return frames_by_participant

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

    def run(self):
        """Main method to process all frames in the folder"""
        print(f"Starting frame analysis with {self.model}...")
        print(f"Frames folder: {self.frames_folder}")
        print(f"Output CSV: {self.output_csv}")
        print(f"Prompt: {self.base_prompt}")

        # Get all frame files organized by participant
        frames_by_participant = self.get_frame_files()

        if not frames_by_participant:
            print(f"No frame files found in {self.frames_folder}")
            print(f"Supported formats: {', '.join(self.image_extensions)}")
            return

        print(f"Found {len(frames_by_participant)} participants")
        total_frames = sum(len(frames) for frames in frames_by_participant.values())
        print(f"Total frames: {total_frames}")

        # Process each participant
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
    parser = argparse.ArgumentParser(description='Frame Analysis using Vision-Language Models')
    parser.add_argument('--model', type=str, default='llama3.2-vision',
                        help='Ollama model to use (default: llama3.2-vision)')
    parser.add_argument('--prompt', type=str,
                        default='Given the human reaction shown in the image, do you think the situation observed by that human ends well or poorly? (Use only one word -- well or poorly -- to answer)',
                        help='Prompt for the vision model')
    parser.add_argument('--frames-folder', type=str, default='./frames',
                        help='Folder containing frame files organized by participant (default: ./frames)')
    parser.add_argument('--output-csv', type=str, default='frame_results.csv',
                        help='Output CSV file name (default: frame_results.csv)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='Ollama API URL (default: http://localhost:11434)')

    args = parser.parse_args()

    analyzer = FrameAnalyzer(
        model=args.model,
        prompt=args.prompt,
        ollama_url=args.ollama_url,
        frames_folder=args.frames_folder,
        output_csv=args.output_csv
    )

    analyzer.run()

    # Example usage:
     #conda activate ollama
    #ollama pull llama3.2-vision

    #DONE
    # python vlm_reactions_frames.py --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/frames_results_1s.csv'
    # python vlm_reactions_frames.py --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/frames_results_3s.csv'
    #nohup python vlm_reactions_frames.py --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/frames_results_1s.csv' > ./logs/vlm_output_frames_1s.log 2>&1 &
    #nohup python vlm_reactions_frames.py --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/frames_results_3s.csv' > ./logs/vlm_output_frames_3s.log 2>&1 &
    #nohup python vlm_reactions_frames.py --model llava --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/results_llava_1s.csv' > ./logs/vlm_output_llava1.log 2>&1 &
    #nohup python vlm_reactions_frames.py --model llava --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_llava_3s.csv' > ./logs/vlm_output_llava3.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model deepseek-ocr:3b --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/results_deepseek-ocr_1s.csv' > ./logs/vlm_output_deepseek_ocr_1s.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model deepseek-ocr:3b --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_deepseek-ocr_3s.csv' > ./logs/vlm_output_deepseek_ocr_3s.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model qwen3-vl --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/results_qwen3_1s.csv' > ./logs/vlm_output_qwen3_1s.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model qwen3-vl --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_qwen3_3s.csv' > ./logs/vlm_output_qwen3_3s.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model gemma3 --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/results_gemma3_1s.csv' > ./logs/vlm_output_gemma3_1s.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model gemma3 --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_gemma3_3s.csv' > ./logs/vlm_output_gemma3_3s.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model llava-llama3 --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/results_llavallama3_1s.csv' > ./logs/vlm_output_llavallama3_1s.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model llava-llama3 --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_llavallama3_3s.csv' > ./logs/vlm_output_llavallama3_3s.log 2>&1 &
    
    

    #TO DO 
    #nohup python vlm_reactions_frames.py  --model mistral-small3.2 --frames-folder '../../../data/new_reactions/image_dataset_1s/' --output-csv './results/results_mistralsmall32_1s.csv' > ./logs/vlm_output_mistralsmall32_1s.log 2>&1 &
    #nohup python vlm_reactions_frames.py  --model mistral-small3.2 --frames-folder '../../../data/new_reactions/image_dataset_3s/' --output-csv './results/results_mistralsmall32_3s.csv' > ./logs/vlm_output_mistralsmall32_3s.log 2>&1 &
   

if __name__ == "__main__":
    main()
