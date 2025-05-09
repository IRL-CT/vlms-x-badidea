import os
import json
import glob
from openpyxl import Workbook
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import cv2

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)

# Set up the prompts
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

# Helper functions from the reference
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, 
               ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=16):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# Generation configuration
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=100,
    top_p=0.1,
    num_beams=1
)

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
                # Load video and process with the proper method
                with torch.no_grad():
                    pixel_values, num_patches_list = load_video(clip_path, num_segments=16, max_num=1)
                    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
                    
                    # Create video prefix
                    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                    
                    # Run prompts
                    for prompt_name, prompt_text in prompts.items():
                        question = video_prefix + prompt_text
                        
                        try:
                            output, _ = model.chat(
                                tokenizer, 
                                pixel_values, 
                                question, 
                                generation_config, 
                                num_patches_list=num_patches_list, 
                                history=None, 
                                return_history=True
                            )
                            ws.append([video_id, start_time, prompt_name, output])
                            print(f"Response: {output}")
                        except Exception as e:
                            ws.append([video_id, start_time, prompt_name, f"ERROR: {str(e)}"])
                            print(f"❌ Error during model generation: {e}")
                
            except Exception as e:
                print(f"Error processing clip {clip_path}: {str(e)}")
        
        # Save progress after each video
        wb.save("video_descriptions_internVideo.xlsx")
    
    except Exception as e:
        print(f"Error processing clip info file {clip_info_file}: {str(e)}")

print("Processing complete. Results saved to video_descriptions_internVideo.xlsx")