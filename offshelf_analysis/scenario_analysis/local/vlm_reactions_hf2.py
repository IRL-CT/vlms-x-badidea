"""
Vision Language Model Demo for Video Analysis - Hugging Face Version (Fixed for older transformers)

This script processes videos from a folder and sends frames to a Vision-Language Model
(VLM) for analysis and outcome prediction. Compatible with transformers 4.28.1.
Results are saved to a CSV file.
"""

import cv2
import time
import argparse
import os
import csv
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer
)

# Try to import newer classes, but don't fail if they're not available
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    HAS_VISION2SEQ = True
except ImportError:
    HAS_VISION2SEQ = False
    print("AutoModelForVision2Seq not available - using BLIP models only")

import warnings
warnings.filterwarnings("ignore")

class HuggingFaceVideoAnalyzer:
    def __init__(self, 
                 model_name="Salesforce/blip-image-captioning-large",
                 prompt="Question: Given the scenario shown in the image, do you think this situation ends well or poorly? Answer:",
                 video_folder="./videos",
                 output_csv="reaction_results.csv",
                 frame_sample_rate=15,
                 device="auto"):
        
        self.model_name = model_name
        self.base_prompt = prompt
        self.video_folder = video_folder
        self.output_csv = output_csv
        self.frame_sample_rate = frame_sample_rate
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Supported video formats
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        # Results storage
        self.results = []
        
        # Model compatibility mapping
        self.compatible_models = {
            # BLIP models that work with older transformers
            "Salesforce/blip-image-captioning-base": "blip_caption",
            "Salesforce/blip-image-captioning-large": "blip_caption", 
            "Salesforce/blip-vqa-base": "blip_vqa",
            "Salesforce/blip-vqa-capfilt-large": "blip_vqa",
        }
        
        # Load model and processor
        self.load_model()
        
    def load_model(self):
        """Load the specified Hugging Face model and processor"""
        print(f"Loading model: {self.model_name}")
        
        # Check if model is in our compatibility list
        if self.model_name not in self.compatible_models:
            print(f"Model {self.model_name} not in compatibility list.")
            print("Available compatible models:")
            for model in self.compatible_models.keys():
                print(f"  - {model}")
            print("Falling back to default BLIP model...")
            self.model_name = "Salesforce/blip-image-captioning-large"
        
        try:
            model_type = self.compatible_models[self.model_name]
            
            if model_type in ["blip_caption", "blip_vqa"]:
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                
                self.model_type = model_type
                print(f"Model loaded successfully on {self.device}")
                print(f"Model type: {model_type}")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to basic BLIP model...")
            self.fallback_to_basic_blip()
            
    def fallback_to_basic_blip(self):
        """Fallback to the most reliable BLIP model"""
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        self.model_type = "blip_caption"
        print(f"Fallback model loaded: {self.model_name}")
    
    def get_video_files(self):
        """Get all video files from the specified folder"""
        video_files = []
        for file_path in Path(self.video_folder).iterdir():
            if file_path.is_dir():
                for sub_file in file_path.rglob('*'):
                    if sub_file.is_file() and sub_file.suffix.lower() in self.video_extensions:
                        if '_30fps' not in sub_file.name:
                            video_files.append(sub_file)
            else:
                if file_path.suffix.lower() in self.video_extensions:
                    if '_30fps' not in file_path.name:
                        video_files.append(file_path)
        return sorted(video_files)
    
    def preprocess_image(self, frame):
        """Convert OpenCV frame to PIL Image and preprocess"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Resize to reasonable size to save memory - BLIP works well with smaller images
        pil_img = pil_img.resize((384, 384))
        
        return pil_img
    
    def format_prompt_for_model(self, prompt):
        """Format prompt based on model type"""
        if self.model_type == "blip_vqa":
            # VQA models expect question format
            if not prompt.lower().startswith("question:"):
                return f"Question: {prompt} Answer:"
            return prompt
        elif self.model_type == "blip_caption":
            # Captioning models - we'll use conditional generation
            return prompt
        else:
            return prompt
    
    def analyze_frame(self, frame, video_name):
        """Analyze frame using Hugging Face model"""
        try:
            # Preprocess image
            image = self.preprocess_image(frame)
            
            # Format prompt based on model type
            formatted_prompt = self.format_prompt_for_model(self.base_prompt)
            
            # Prepare inputs based on model type
            if self.model_type == "blip_vqa":
                # For VQA models, use the question format
                inputs = self.processor(image, formatted_prompt, return_tensors="pt")
            elif self.model_type == "blip_caption":
                # For captioning models, we can try conditional generation
                # Some BLIP captioning models support text input for conditioning
                try:
                    inputs = self.processor(image, formatted_prompt, return_tensors="pt")
                except:
                    # If text input not supported, just use image
                    inputs = self.processor(image, return_tensors="pt")
            else:
                inputs = self.processor(image, formatted_prompt, return_tensors="pt")
            
            # Move inputs to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                if self.model_type == "blip_vqa":
                    # VQA models
                    generate_ids = self.model.generate(
                        **inputs,
                        max_length=30,
                        min_length=2,
                        num_beams=3,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )
                elif self.model_type == "blip_caption":
                    # Captioning models
                    if 'input_ids' in inputs:
                        # Conditional captioning
                        generate_ids = self.model.generate(
                            **inputs,
                            max_length=50,
                            min_length=5,
                            num_beams=3,
                            early_stopping=True,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.pad_token_id
                        )
                    else:
                        # Regular captioning
                        generate_ids = self.model.generate(
                            inputs['pixel_values'],
                            max_length=50,
                            min_length=5,
                            num_beams=3,
                            early_stopping=True,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.pad_token_id
                        )
                else:
                    # Generic generation
                    generate_ids = self.model.generate(
                        **inputs,
                        max_length=30,
                        num_beams=3,
                        early_stopping=True,
                        do_sample=False
                    )
                
                generated_text = self.processor.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
            
            # Clean up the response
            response = generated_text.strip()
            
            # Post-process response based on model type
            if self.model_type == "blip_vqa":
                # For VQA, remove question if included
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                elif "Question:" in response:
                    # Sometimes the whole prompt is repeated
                    parts = response.split("Answer:")
                    if len(parts) > 1:
                        response = parts[-1].strip()
                    else:
                        # Extract after question
                        if formatted_prompt in response:
                            response = response.replace(formatted_prompt, "").strip()
            
            elif self.model_type == "blip_caption":
                # For captioning, the response might include our conditioning text
                if formatted_prompt in response:
                    response = response.replace(formatted_prompt, "").strip()
            
            # Extract key sentiment words if the response is long
            if len(response.split()) > 3:
                response_words = response.lower().split()
                sentiment_words = ['well', 'poorly', 'good', 'bad', 'positive', 'negative', 
                                'success', 'successful', 'failure', 'fail', 'happy', 'sad',
                                'favorable', 'unfavorable', 'excellent', 'terrible']
                
                for word in response_words:
                    clean_word = word.strip('.,!?')
                    if clean_word in sentiment_words:
                        return clean_word
                
                # If no sentiment word found, return first few words
                return ' '.join(response_words[:2])
            
            return response.lower().strip('.,!?')
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def process_video(self, video_path):
        """Process a single video file"""
        video_name = video_path.name
        print(f"\nProcessing video: {video_name}")
        video_folder = video_path.parent.name
        
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
        participant = []
        
        for i, frame in enumerate(sampled_frames):
            print(f"Analyzing frame {i+1}/{len(sampled_frames)}...")
            response = self.analyze_frame(frame, video_name)
            responses.append(response)
            print(f"Response: {response}")
            video_names.append(video_name)
            participant.append(video_folder)
            
            # Clear GPU cache to prevent memory issues
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Small delay to prevent overheating
            time.sleep(0.1)
        
        return video_names, participant, frame_times, responses
    
    def save_results_to_csv(self):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save.")
            return
            
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_name', 'participant', 'frame', 'outcome_prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        
        print(f"\nResults saved to {self.output_csv}")
        print(f"Total frames processed: {len(self.results)}")
    
    def run(self):
        """Main method to process all videos in the folder"""
        print(f"Starting video analysis with {self.model_name}...")
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
                    # Append results to the list
                    for j in range(len(result[0])):
                        self.results.append({
                            'video_name': result[0][j],
                            'participant': result[1][j],
                            'frame': result[2][j],
                            'outcome_prediction': result[3][j]
                        })
                    print(f"Final prediction for {video_path.name}: {result[3][0]}")
                else:
                    print(f"Failed to process {video_path.name}")

                # Save results to CSV after each video
                self.save_results_to_csv()
                    
            except Exception as e:
                print(f"Error processing {video_path.name}: {str(e)}")
                self.results.append({
                    'video_name': video_path.name,
                    'participant': 'unknown',
                    'frame': 0,
                    'outcome_prediction': f"Error: {str(e)}"
                })
        
        # Final save
        self.save_results_to_csv()
        
        print(f"\n{'='*50}")
        print("Analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Video Analysis using Hugging Face Vision-Language Models')
    parser.add_argument('--model', type=str, default='Salesforce/blip-vqa-base', 
                        help='Hugging Face model to use (compatible models only)')
    parser.add_argument('--prompt', type=str, 
                        default='Given the scenario shown in the image, do you think this situation ends well or poorly?',
                        help='Prompt for the vision model')
    parser.add_argument('--video-folder', type=str, default='./videos',
                        help='Folder containing video files')
    parser.add_argument('--output-csv', type=str, default='results.csv',
                        help='Output CSV file name')
    parser.add_argument('--frame-sample-rate', type=int, default=15,
                        help='Sample every Nth frame from video')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Create video folder if it doesn't exist
    os.makedirs(args.video_folder, exist_ok=True)
    
    analyzer = HuggingFaceVideoAnalyzer(
        model_name=args.model,
        prompt=args.prompt,
        video_folder=args.video_folder,
        output_csv=args.output_csv,
        frame_sample_rate=args.frame_sample_rate,
        device=args.device
    )
    
    analyzer.run()

if __name__ == "__main__":
    main()

# Compatible models for transformers 4.28.1:
# python vlm_reactions_hf.py --model "Salesforce/blip-vqa-base"
# python vlm_reactions_hf.py --model "Salesforce/blip-vqa-capfilt-large"  
# python vlm_reactions_hf.py --model "Salesforce/blip-image-captioning-base"
# python vlm_reactions_hf.py --model "Salesforce/blip-image-captioning-large"

# Example with your data:
# python vlm_reactions_hf.py --video-folder "../../../../data/final_cut_videos/" --model "Salesforce/blip-vqa-base" --output-csv "./results/results_blip_vqa.csv"