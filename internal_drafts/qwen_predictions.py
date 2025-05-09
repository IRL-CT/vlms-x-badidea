import os
import json
import glob
from openpyxl import Workbook
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import cv2
# Load the model and processor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # 在 GPU 上使用 float16 以节省内存
    device_map="auto" if device == "cuda" else {"": "cpu"},
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Set up the prompts as specified
prompts = {
    "Prompt 5.1": "Given the scenario shown on the video, You think this situation ends well or poorly? (Use only one word to answer)",
    "Prompt 6.1": "Given the scenario shown on the video, You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)",
}

# Create Excel workbook
wb = Workbook()
ws = wb.active
ws.title = "Video Descriptions"
# Add headers
ws.append(["VIDEO", "TIME_START", "PROMPT", "DESCRIPTION"])

# Extract frames from video
def extract_frames(video_path, output_dir, max_frames=10):  # 限制最大帧数
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算采样间隔
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
            frame_paths.append(f"file://{os.path.abspath(frame_file)}")
            saved += 1
        i += 1
    cap.release()
    
    print(f"Extracted {len(frame_paths)} frames")
    return frame_paths

# Function to get description using the model for video
def get_video_description(image_paths, prompt_text):
    # Create messages in the format expected by Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": image_paths,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    
    # Create model inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    
    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
    
    # Decode only the new tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()

# Find all clip info JSON files
clip_info_files = glob.glob("./clips/*_clips_info.json")

# Process each clip info file
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
                # Step 1: Extract images
                frame_dir = f"./tmp_frames/{video_id}_{os.path.basename(clip_path).replace('.mp4','')}"
                frame_paths = extract_frames(clip_path, frame_dir, max_frames=6)

                if not frame_paths:
                    print(f"⚠️  No frames extracted from {clip_path}")
                    continue

                # Step 2: Run prompts
                for prompt_name, prompt_text in prompts.items():
                    try:
                        description = get_video_description(frame_paths, prompt_text)
                        ws.append([video_id, start_time, prompt_name, description])
                    except Exception as e:
                        ws.append([video_id, start_time, prompt_name, f"ERROR: {str(e)}"])
                        print(f"❌ Error during model generation: {e}")
            except Exception as e:
                print(f"Error processing clip {clip_path}: {str(e)}")
        
        # Save progress after each video
        wb.save("video_descriptions.xlsx")
    except Exception as e:
        print(f"Error processing clip info file {clip_info_file}: {str(e)}")

print("Processing complete. Results saved to video_descriptions.xlsx")