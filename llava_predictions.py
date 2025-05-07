import os
import json
import glob
import cv2
import requests
import base64
from tqdm import tqdm
from openpyxl import Workbook

# 设置Ollama API的URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# 创建Excel工作簿
wb = Workbook()
ws = wb.active
ws.title = "Video Descriptions"
# 添加表头
ws.append(["VIDEO", "TIME_START", "PROMPT", "DESCRIPTION"])

# 设置提示词
prompts = {
    "Prompt 5.1": "Given the scenario shown on the video, You think this situation ends well or poorly? (Use only one word to answer)",
    "Prompt 6.1": "Given the scenario shown on the video, You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)",
}

# 从视频提取帧
def extract_frames(video_path, output_dir, max_frames=6):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算采样间隔
    every_n_frames = max(1, total_frames // max_frames)
    
    i = 0
    saved = 0
    frame_paths = []
    
    print(f"从 {video_path} 提取帧，每 {every_n_frames} 帧采样一次...")
    
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
    
    print(f"提取了 {len(frame_paths)} 帧")
    return frame_paths

# 使用LLaVA模型获取视频描述
def get_video_description(frame_paths, prompt_text):
    # 读取第一帧图像并转换为base64
    with open(frame_paths[0], "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
    
    # 构建API请求
    payload = {
        "model": "llava",
        "prompt": prompt_text,
        "images": [img_base64],
        "stream": False
    }
    
    try:
        # 发送请求到Ollama API
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # 如果有多帧，我们可以继续询问模型关于后续帧的看法
        if len(frame_paths) > 1:
            additional_context = result.get("response", "")
            # 处理剩余帧
            for i in range(1, min(3, len(frame_paths))):  # 只处理前3帧以避免过长
                with open(frame_paths[i], "rb") as img_file:
                    next_img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                
                additional_prompt = f"Here's another frame from the same video. Based on this and previous frames, {prompt_text}"
                
                next_payload = {
                    "model": "llava",
                    "prompt": additional_prompt,
                    "images": [next_img_base64],
                    "stream": False
                }
                
                next_response = requests.post(OLLAMA_API_URL, json=next_payload)
                next_response.raise_for_status()
                next_result = next_response.json()
                additional_context += " " + next_result.get("response", "")
            
            # 最终综合分析
            final_prompt = f"Based on all the frames you've seen from this video, {prompt_text}"
            
            final_payload = {
                "model": "llava",
                "prompt": final_prompt,
                "stream": False
            }
            
            final_response = requests.post(OLLAMA_API_URL, json=final_payload)
            final_response.raise_for_status()
            final_result = final_response.json()
            
            # 返回最终回答
            return final_result.get("response", "")
        
        # 如果只有一帧，直接返回结果
        return result.get("response", "")
    
    except Exception as e:
        print(f"API请求错误: {e}")
        return f"错误: {str(e)}"

# 处理每个clip信息文件
clip_info_files = glob.glob("./clips/*_clips_info.json")

for clip_info_file in tqdm(clip_info_files, desc="处理视频"):
    try:
        with open(clip_info_file, 'r') as f:
            clips_info = json.load(f)
        
        # 处理每个clip
        for clip_info in tqdm(clips_info, desc=f"处理 {os.path.basename(clip_info_file)} 中的clips", leave=False):
            print(f"📦 正在处理clip信息文件: {clip_info_file}", flush=True)
            video_id = clip_info["video_id"]
            clip_path = clip_info["clip_path"]
            start_time = clip_info["start_time"]
            
            # 检查clip文件是否存在
            if not os.path.exists(clip_path):
                print(f"警告: Clip文件 {clip_path} 不存在。跳过。")
                continue
            
            try:
                # 提取帧
                frame_dir = f"./tmp_frames/{video_id}_{os.path.basename(clip_path).replace('.mp4','')}"
                frame_paths = extract_frames(clip_path, frame_dir, max_frames=6)
                
                if not frame_paths:
                    print(f"⚠️ 从 {clip_path} 没有提取到帧")
                    continue
                
                # 对每个提示词，获取视频描述
                for prompt_name, prompt_text in prompts.items():
                    try:
                        # 获取描述
                        description = get_video_description(frame_paths, prompt_text)
                        ws.append([video_id, start_time, prompt_name, description])
                    except Exception as e:
                        ws.append([video_id, start_time, prompt_name, f"错误: {str(e)}"])
                        print(f"❌ 模型生成过程中出错: {e}")
            except Exception as e:
                print(f"处理clip {clip_path} 时出错: {str(e)}")
        
        # 每处理完一个视频保存进度
        wb.save("video_descriptions_llava.xlsx")
    except Exception as e:
        print(f"处理clip信息文件 {clip_info_file} 时出错: {str(e)}")

print("处理完成。结果已保存到 video_descriptions_llava.xlsx")