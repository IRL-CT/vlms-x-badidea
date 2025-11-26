"""
Vision Language Model Demo for Video Analysis - Hugging Face Version

This script processes videos from a folder and sends frames to a Vision-Language Model
(VLM) for analysis and outcome prediction. The system uses Hugging Face transformers
to access open-source multimodal models for local inference.
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
    AutoProcessor, 
    AutoModelForVision2Seq,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer
)
import warnings
warnings.filterwarnings("ignore")

class HuggingFaceVideoAnalyzer:
    def __init__(self, 
                 model_name="llava-hf/llava-v1.6-mistral-7b-hf",
                 prompt="Given the scenario shown in the image, do you think this situation ends well or poorly? Answer with only one word.",
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
        
        # Load model and processor
        self.load_model()
        
    def load_model(self):
        """Load the specified Hugging Face model and processor"""
        print(f"Loading model: {self.model_name}")
        
        try:
            # Handle different model types
            if "llava" in self.model_name.lower():
                if "next" in self.model_name.lower() or "v1.6" in self.model_name.lower():
                    self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        self.model_name, 
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                else:
                    # For older LLaVA models
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
            elif "blip" in self.model_name.lower():
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
            else:
                # Generic approach for other models
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to BLIP-2 model...")
            self.model_name = "Salesforce/blip2-opt-2.7b"
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
    
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
        
        # Resize to reasonable size to save memory
        pil_img = pil_img.resize((512, 512))
        
        return pil_img
    
    def analyze_frame(self, frame, video_name):
        """Analyze frame using Hugging Face model"""
        try:
            # Preprocess image
            image = self.preprocess_image(frame)
            
            # Prepare inputs based on model type
            if "llava" in self.model_name.lower():
                # LLaVA models expect conversation format
                if "next" in self.model_name.lower() or "v1.6" in self.model_name.lower():
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.base_prompt},
                                {"type": "image"},
                            ],
                        },
                    ]
                    prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                else:
                    prompt = f"USER: <image>\n{self.base_prompt}\nASSISTANT:"
                    inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            elif "blip" in self.model_name.lower():
                # BLIP models
                inputs = self.processor(images=image, text=self.base_prompt, return_tensors="pt")
            
            else:
                # Generic approach
                inputs = self.processor(images=image, text=self.base_prompt, return_tensors="pt")
            
            # Move inputs to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                if "llava" in self.model_name.lower():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                        temperature=0.2,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    # Decode only the generated part
                    generated_text = self.processor.batch_decode(
                        generate_ids[:, inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
                else:
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                        temperature=0.2,
                        pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                    )
                    generated_text = self.processor.batch_decode(
                        generate_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
            
            # Clean up the response
            response = generated_text.strip()
            
            # For LLaVA models, remove the prompt part if it's included
            if "llava" in self.model_name.lower() and self.base_prompt in response:
                response = response.split(self.base_prompt)[-1].strip()
            
            return response
            
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
    parser.add_argument('--model', type=str, default='llava-hf/llava-v1.6-mistral-7b-hf', 
                        help='Hugging Face model to use')
    parser.add_argument('--prompt', type=str, 
                        default='Given the scenario shown in the image, do you think this situation ends well or poorly? Answer with only one word.',
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

# Example usage with different models:
# python vlm_reactions_hf.py --video-folder "../../../../data/final_cut_videos/" --model "llava-hf/llava-v1.6-mistral-7b-hf" --output-csv "results_llava16.csv"
# python vlm_reactions_hf.py --video-folder "../../../../data/final_cut_videos/" --model "Salesforce/blip2-opt-2.7b" --output-csv "results_blip2.csv"
# python vlm_reactions_hf.py --video-folder "../../../../data/final_cut_videos/" --model "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" --output-csv "results_qwen20.csv"


# python hf_video_analyzer.py --model "llava-hf/llava-1.5-7b-hf"
# python hf_video_analyzer.py --model "microsoft/kosmos-2-patch14-224"
# python hf_video_analyzer.py --model "Salesforce/instructblip-vicuna-7b"



#python vlm_reactions_hf.py --video-folder "../../../../data/final_cut_videos/" --model "llava-hf/llava-v1.6-mistral-7b-hf" --output-csv "./results/results_llava16.csv"
#python vlm_reactions_hf.py --video-folder "../../../../data/final_cut_videos/" --model "llava-hf/llava-v1.6-vicuna-7b-hf" --output-csv "./results/results_llava16vicuna.csv"
#python vlm_reactions_hf.py --video-folder "../../../../data/final_cut_videos/" --model "google/gemma-3n-E4B-it" --output-csv "./results/results_gemma3ne4b.csv"
#python vlm_reactions_hf.py --video-folder "../../../../data/final_cut_videos/" --model "Qwen/Qwen2.5-VL-7B-Instruct" --output-csv "./results/result_qwen25.csv"