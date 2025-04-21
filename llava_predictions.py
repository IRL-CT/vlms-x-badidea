import os
import json
import glob
from openpyxl import Workbook
import torch
import copy
import warnings
import gc
from tqdm import tqdm
from decord import VideoReader, cpu
import numpy as np
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")

# Import LLaVA modules
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

# Function to load and process video frames
def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps_step = round(vr.get_avg_fps()/fps)
    
    frame_idx = [i for i in range(0, len(vr), fps_step)]
    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    return spare_frames, frame_time_str, video_time

# Load the model
print("Loading LLaVA-Video-7B-Qwen2 model...")
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

try:
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, 
        None, 
        model_name, 
        torch_dtype="float16",  # 明确使用 torch.float16 而不是字符串
        device_map=device_map,
        attn_implementation="eager",
        use_flash_attention_2=False,
    )
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Set up the prompts
prompts = {
    "Prompt 5.1": "Describe what you see in this video",
    # "Prompt 6.1": "Given the scenario shown on the video, You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)",
}

# Create Excel workbook
wb = Workbook()
ws = wb.active
ws.title = "Video Descriptions"
# Add headers
ws.append(["VIDEO", "TIME_START", "PROMPT", "DESCRIPTION"])

# Function to process a video with the LLaVA model
# def process_video_with_llava(video_path, prompt_text, max_frames=8):
#     try:
#         # Load and process video frames
#         video_frames, frame_time, video_time = load_video(
#             video_path, 
#             max_frames_num=max_frames, 
#             fps=1, 
#             force_sample=True
#         )
        
#         # Process video frames with image processor
#         processed_video = image_processor.preprocess(
#             video_frames, 
#             return_tensors="pt"
#         )["pixel_values"].to(device).to(torch.float16)
        
#         video = [processed_video]
        
#         # Set up conversation template
#         conv_template = "qwen_1_5"
#         time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
        
#         # Format the question with the prompt
#         question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt_text}"
        
#         # Create conversation
#         conv = copy.deepcopy(conv_templates[conv_template])
#         conv.append_message(conv.roles[0], question)
#         conv.append_message(conv.roles[1], None)
#         prompt_question = conv.get_prompt()
        
#         # Tokenize input
#         input_ids = tokenizer_image_token(
#             prompt_question, 
#             tokenizer, 
#             IMAGE_TOKEN_INDEX, 
#             return_tensors="pt"
#         ).unsqueeze(0).to(device)
        
#         # Generate response
#         with torch.no_grad():
#             output_ids = model.generate(
#                 input_ids,
#                 images=video,
#                 modalities=["video"],
#                 do_sample=False,
#                 num_beams=1,      # 使用简单的贪婪搜索
#                 temperature=1.0,  # 中性温度
#                 max_new_tokens=100,
#             )
        
#         # Decode output
#         output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
#         processed_response = output_text.lower()
#         if "well" in processed_response:
#             return "well"
#         elif "poor" in processed_response or "poorly" in processed_response:
#             return "poorly"
#         else:
#             return processed_response[:30]  # 返回更长的截断回答
                
#     except Exception as e:
#         return f"ERROR: {str(e)}"
#     finally:
#         # Clean up memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()
# 确保每次只处理一个视频（批处理尺寸=1）
def process_video_with_llava(video_path, prompt_text, max_frames=4):
    try:
        # 加载和处理视频帧
        video_frames, frame_time, video_time = load_video(
            video_path, 
            max_frames_num=max_frames, 
            fps=1, 
            force_sample=True
        )
        
        # 处理单个视频
        processed_video = image_processor.preprocess(
            video_frames, 
            return_tensors="pt"
        )["pixel_values"].to(device).to(torch.float16)
        
        # 注意这里只使用一个视频
        video = [processed_video]
        
        # 设置会话模板
        conv_template = "qwen_1_5"
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
        
        # 格式化问题
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt_text}"
        print(question)
        # 创建会话
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        print(prompt_question)
        # 标记化输入（一次只处理一个）
        input_ids = tokenizer_image_token(
            prompt_question, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).to(device)
        print("EOS Token ID:", tokenizer.eos_token_id)
        print("PAD Token ID:", tokenizer.pad_token_id)
        # 打印 tokenizer 的一些基本信息
        print("Vocab size:", tokenizer.vocab_size)
        print("Special tokens:", tokenizer.all_special_tokens)
        print("Special token IDs:", tokenizer.all_special_ids)
        
        exclamation_token_id = tokenizer.encode("!", add_special_tokens=False)[0]
        print(f"'!' token ID: {exclamation_token_id}")
        # 生成响应
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=100,
            )
        
        # 解码输出
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print("Image token:", tokenizer.convert_ids_to_tokens([IMAGE_TOKEN_INDEX]))
        print("Special tokens:", tokenizer.special_tokens_map)
        print(output_text)
        
        # 提取模型的响应（在提示之后）
        if conv.roles[1] in output_text:
            response = output_text.split(conv.roles[1])[-1].strip()
        else:
            response = output_text.strip()
        
        # 处理回答格式
        response = response.lower()
        # if "well" in response:
        #     return "well"
        # elif "poor" in response:
        #     return "poorly"
        # else:
        return response  # 截断长回答
        
    except Exception as e:
        return f"ERROR: {str(e)}"
    finally:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
# Process all clip info files
clip_info_files = glob.glob("./clips/*_clips_info.json")

for clip_info_file in tqdm(clip_info_files, desc="Processing videos"):
    try:
        with open(clip_info_file, 'r') as f:
            clips_info = json.load(f)
        
        # Process each clip
        for clip_idx, clip_info in enumerate(tqdm(clips_info, desc=f"Processing {os.path.basename(clip_info_file)}")):
            video_id = clip_info["video_id"]
            clip_path = clip_info["clip_path"]
            start_time = clip_info["start_time"]
            
            # Check if the clip file exists
            if not os.path.exists(clip_path):
                print(f"Warning: Clip file {clip_path} does not exist. Skipping.")
                continue
            
            # Process each prompt
            for prompt_name, prompt_text in prompts.items():
                try:
                    # Use fewer frames for memory efficiency
                    result = process_video_with_llava(clip_path, prompt_text, max_frames=4)
                    ws.append([video_id, start_time, prompt_name, result])
                except Exception as e:
                    error_msg = f"ERROR: {str(e)}"
                    print(f"Error processing {prompt_name} for {clip_path}: {error_msg}")
                    ws.append([video_id, start_time, prompt_name, error_msg])
            
            # Save progress regularly
            if clip_idx % 3 == 0:  # Save more frequently due to potential memory issues
                wb.save("video_descriptions_llava.xlsx")
                
            # Clear memory after each clip
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error processing clip info file {clip_info_file}: {str(e)}")

# Final save
wb.save("video_descriptions_llava.xlsx")
print("Processing complete. Results saved to video_descriptions_llava.xlsx")