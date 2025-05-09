import os
import json
import glob
import torch
from tqdm import tqdm
import cv2
from openpyxl import Workbook

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create Excel workbook
wb = Workbook()
ws = wb.active
ws.title = "Video Descriptions"
# Add headers
ws.append(["VIDEO", "TIME_START", "PROMPT", "DESCRIPTION"])

# Set up the prompts as specified
prompts = {
    "Prompt 5.1": "Given the scenario shown on the video, You think this situation ends well or poorly? (Use only one word to answer)",
    "Prompt 6.1": "Given the scenario shown on the video, You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)",
}

# Create folder for extracted frames
os.makedirs("./tmp_frames", exist_ok=True)

# Extract frames from video function
def extract_frames(video_path, output_dir, max_frames=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sampling interval
    every_n_frames = max(1, total_frames // max_frames)
    
    i = 0
    saved = 0
    frame_paths = []
    
    print(f"Extracting frames from {video_path}, every {every_n_frames} frames...")
    
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % every_n_frames == 0:
            frame_file = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_file, frame)
            frame_paths.append(os.path.abspath(frame_file))
            saved += 1
        i += 1
    cap.release()
    
    print(f"Extracted {len(frame_paths)} frames")
    return frame_paths

# This function will be implemented once we load the model
def lazy_load_model():
    """Lazy loading of the model to avoid importing it at the top"""
    try:
        # First try to import with the recommended transformers version
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        # Load VideoLLaMA3 model and processor
        model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Loading model from {model_path}...")
        
        # Try loading with flash_attention first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else {"": "cpu"},
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            print("Successfully loaded model with flash_attention_2")
        except Exception as e:
            print(f"Failed to load with flash_attention_2: {e}")
            print("Trying to load without specific attention implementation...")
            
            # Try loading without specific attention implementation
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else {"": "cpu"},
                torch_dtype=torch_dtype,
            )
            print("Successfully loaded model without specific attention implementation")
        
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor
    
    except Exception as e:
        print(f"Error loading VideoLLaMA3: {e}")
        print("Please ensure you have the required dependencies:")
        print("pip install transformers==4.46.3 accelerate==1.0.1")
        print("pip install flash-attn --no-build-isolation")
        return None, None

# Function to get description using VideoLLaMA3
def get_video_description(model, processor, video_path, prompt_text, max_frames=6, fps=1):
    if model is None or processor is None:
        return f"ERROR: Model not loaded properly. Please check dependencies."
    
    try:
        # For VideoLLaMA3, we first try the direct video approach
        print(f"Attempting to process video directly: {video_path}")
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": {
                            "video_path": video_path,
                            "fps": fps,
                            "max_frames": max_frames
                        }
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        with torch.inference_mode():
            # Process conversation for model input
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Convert pixel values to proper dtype if needed
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)
            
            # Generate output
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
            
            # Decode the response
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Extract only the assistant's response
            assistant_response = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response
            
            return assistant_response
    
    except Exception as e:
        print(f"Error in direct video processing: {str(e)}")
        
        # Fall back to frame extraction approach
        print(f"Falling back to frame extraction approach...")
        
        try:
            # Extract frames
            frame_dir = f"./tmp_frames/{os.path.basename(video_path).replace('.mp4','')}"
            frame_paths = extract_frames(video_path, frame_dir, max_frames=max_frames)
            
            if not frame_paths:
                return f"ERROR: No frames could be extracted from {video_path}"
                
            # Create a conversation with individual frames
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"These images are frames from a video. {prompt_text}"}
                    ],
                }
            ]
            
            # Add frames to the conversation content
            for frame_path in frame_paths:
                conversation[1]["content"].insert(0, {
                    "type": "image", 
                    "image": {"image_path": frame_path}
                })
            
            with torch.inference_mode():
                # Process and generate as before
                inputs = processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)
                
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                )
                
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                assistant_response = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response
                
                return assistant_response
                
        except Exception as nested_e:
            return f"ERROR: Both direct video and frame extraction methods failed. Original error: {str(e)}. Frame extraction error: {str(nested_e)}"

# Main processing function
def process_clips():
    # Lazy load model when needed
    model, processor = lazy_load_model()
    
    # Process each clip info file
    clip_info_files = glob.glob("./clips/*_clips_info.json")
    
    for clip_info_file in tqdm(clip_info_files, desc="Processing videos"):
        try:
            with open(clip_info_file, 'r') as f:
                clips_info = json.load(f)
            
            # Process each clip
            for clip_info in tqdm(clips_info, desc=f"Processing clips in {os.path.basename(clip_info_file)}", leave=False):
                print(f"📦 Now processing clip info file: {clip_info_file}", flush=True)
                video_id = clip_info["video_id"]
                clip_path = clip_info["clip_path"]
                start_time = clip_info["start_time"]
                
                # Check if the clip file exists
                if not os.path.exists(clip_path):
                    print(f"Warning: Clip file {clip_path} does not exist. Skipping.")
                    continue
                
                try:
                    # For each prompt, get the video description
                    for prompt_name, prompt_text in prompts.items():
                        try:
                            # Get description
                            description = get_video_description(
                                model=model,
                                processor=processor,
                                video_path=clip_path,
                                prompt_text=prompt_text,
                                max_frames=6,
                                fps=1
                            )
                            ws.append([video_id, start_time, prompt_name, description])
                        except Exception as e:
                            ws.append([video_id, start_time, prompt_name, f"ERROR: {str(e)}"])
                            print(f"❌ Error during model generation: {e}")
                except Exception as e:
                    print(f"Error processing clip {clip_path}: {str(e)}")
            
            # Save progress after each video
            wb.save("video_descriptions_videollama.xlsx")
        except Exception as e:
            print(f"Error processing clip info file {clip_info_file}: {str(e)}")

if __name__ == "__main__":
    process_clips()
    print("Processing complete. Results saved to video_descriptions_videollama.xlsx")