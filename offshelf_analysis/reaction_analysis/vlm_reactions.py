"""
Vision Language Model Demo for Video Analysis

This script processes videos from a folder and sends frames to a Vision-Language Model
(VLM) for analysis and outcome prediction. The system uses Ollama to access
open-source multimodal models like Llama 3.2 Vision for completely local inference.
Results are saved to a CSV file.
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

class VideoAnalyzer:
    def __init__(self, 
                 model="llama3.2-vision", 
                 prompt="Given the reaction shown on the video, you think this situation ends well or poorly? (Use only one word to answer)",
                 ollama_url="http://localhost:11434",
                 video_folder="./videos",
                 output_csv="reaction_results.csv",
                 frame_sample_rate=30):  # Sample every 30th frame
        
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
            #prompt = f"Video: {video_name}. {self.base_prompt}"
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
        """Process a single video file"""
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
        
        # Sample frames from the video
        sampled_frames = []
        frame_indices = range(0, total_frames, self.frame_sample_rate)
        
        frame_times = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                sampled_frames.append(frame)
                frame_times.append(frame_idx)
            
            if len(sampled_frames) >= 45:  # Limit to 45 frames per video
                break
        
        cap.release()
        
        if not sampled_frames:
            print(f"Error: Could not extract frames from {video_name}")
            return None
        
        print(f"Extracted {len(sampled_frames)} frames for analysis")
        
        # Analyze frames and collect responses
        responses = []
        video_names = []
        for i, frame in enumerate(sampled_frames):
            print(f"Analyzing frame {i+1}/{len(sampled_frames)}...")
            response = self.analyze_frame(frame, video_name)
            responses.append(response)
            print(f"Response: {response}")
            video_names.append(video_name)
            time.sleep(1)  # Small delay to avoid overwhelming the API
            #if i >= 2:
            #    break  # For testing, limit to first 3 frames
        
        # Combine responses or take the most common one
        # For now, we'll take the first valid response
        
        #for response in responses:
        #    if not response.startswith("Error"):
        #        final_response.append(response)
        #        video_names.append(video_name)
        #    else:
        #        print(f"Skipping error response: {response}")
        #        break
        
        #if final_response is None:
        #    final_response = responses[0] if responses else "No response"
        
        return video_names, frame_times, responses
    
    def save_results_to_csv(self):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save.")
            return
            
        #create csv file and save results, knowing that it's 3 columns, video_name, frame_time, outcome_prediction
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_name','frame', 'outcome_prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        
        print(f"\nResults saved to {self.output_csv}")
        print(f"Total videos processed: {len(self.results)}")
    
    def run(self):
        """Main method to process all videos in the folder"""
        print(f"Starting video analysis with {self.model}...")
        print(f"Video folder: {self.video_folder}")
        print(f"Output CSV: {self.output_csv}")
        print(f"Prompt: {self.base_prompt}")
        
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
                    #self.results.append({
                    #    'video_name': result[0][0],  # video name
                    #    'frame': result[1],  # frame times
                    #    'outcome_prediction': result
                    #})
                    # Append results to the list
                    for j in range(len(result[0])):
                        self.results.append({
                            'video_name': result[0][j],  # video name
                            'frame': result[1][j],  # frame times
                            'outcome_prediction': result[2][j]  # outcome prediction
                        })
                    print(f"Final prediction for {video_path.name}: {result[2][0]}")
                else:
                    print(f"Failed to process {video_path.name}")

                #save results to CSV after each video
                self.save_results_to_csv()

                    
            except Exception as e:
                print(f"Error processing {video_path.name}: {str(e)}")
                self.results.append({
                    'video_name': video_path.name,
                    'outcome_prediction': f"Error: {str(e)}"
                })
        
        # Save results to CSV
        self.save_results_to_csv()
        
        print(f"\n{'='*50}")
        print("Analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Video Analysis using Vision-Language Models')
    parser.add_argument('--model', type=str, default='llama3.2-vision', 
                        help='Ollama model to use (default: llama3.2-vision)')
    parser.add_argument('--prompt', type=str, 
                        default='Given the scenario shown on the video, you think this situation ends well or poorly? (Use only one word to answer)',
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

    #conda activate ollama
    #ollama pull llama3.2-vision
    #python vlm_reactions.py --video-folder '../../../data/final_cut_videos/' --output-csv './test_results.csv' --frame-sample-rate 15
    #nohup python vlm_reactions.py  --model llava --video-folder '../../../data/final_cut_videos/' --output-csv './results_llava.csv' --frame-sample-rate 15 > vlm_output_llava.log 2>&1 &
    #nohup python vlm_reactions.py --video-folder '../../../data/final_cut_videos/' --output-csv './test_results.csv' --frame-sample-rate 15 > vlm_output.log 2>&1 &
    #nohup python vlm_reactions.py  --model gemma3 --video-folder '../../../data/final_cut_videos/' --output-csv './results_gemma3.csv' --frame-sample-rate 15 > vlm_output_gemma3.log 2>&1 &
    #nohup python vlm_reactions.py  --model gemma3:27b --video-folder '../../../data/final_cut_videos/' --output-csv './results_gemma3_27b.csv' --frame-sample-rate 15 > vlm_output_gemma3S_27b.log 2>&1 &
    #nohup python vlm_reactions.py  --model llama4:scout --video-folder '../../../data/final_cut_videos/' --output-csv './results_llama4.csv' --frame-sample-rate 15 > vlm_output_llama4.log 2>&1 &
    #nohup python vlm_reactions.py  --model qwen2.5vl --video-folder '../../../data/final_cut_videos/' --output-csv './results_qwen25.csv' --frame-sample-rate 15 > vlm_output_qwen25.log 2>&1 &




if __name__ == "__main__":
    main()