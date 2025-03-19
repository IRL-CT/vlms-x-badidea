import os
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Initialize Vertex AI
def init_vertex_ai(project_id="gen-lang-client-0087070594", location="us-central1"):
    vertexai.init(project=project_id, location=location)
    return GenerativeModel("gemini-2.0-flash")

def list_all_clips_info_files(base_path="gs://vlm-testing-vertex-ai/clips/"):
    """List all clips_info.json files in the bucket"""
    cmd = ["gsutil", "ls", f"{base_path}*clips_info.json"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to list clips_info.json files in {base_path}")
    
    # Get all clips_info.json files
    info_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return info_files

def download_json_from_gcs(json_uri, local_path):
    """Download a JSON file from GCS"""
    cmd = ["gsutil", "cp", json_uri, local_path]
    subprocess.run(cmd, check=True)
    return local_path

def get_description(vision_model, clip_uri, retry_count=3):
    """Get description using Vertex AI with Prompt 2"""
    prompt = "Summarize this video as if explaining it to someone who cannot see it. Focus on key actions, emotions, and interactions. Use clear and vivid language to create a mental picture."
    
    for attempt in range(retry_count):
        try:
            response = vision_model.generate_content([
                Part.from_uri(clip_uri, mime_type="video/mp4"),
                Part.from_text(prompt)
            ])
            return response.text
        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                print(f"Error on attempt {attempt+1} for {os.path.basename(clip_uri)}, retrying in {wait_time}s... Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"Failed after {retry_count} attempts for {os.path.basename(clip_uri)}: {e}")
                return f"Error: {str(e)[:100]}..."

def process_single_video(clips_info_uri, output_csv, vision_model, max_workers=5):
    """Process clips for a single video"""
    # Create a temp directory for this video's json file
    video_id = os.path.basename(clips_info_uri).split('_clips_info.json')[0]
    local_json_path = f"/tmp/{video_id}_clips_info.json"
    
    print(f"\nProcessing video: {video_id}")
    print(f"Downloading clips info from {clips_info_uri}")
    
    # Download the clips_info.json file
    download_json_from_gcs(clips_info_uri, local_json_path)
    
    # Load clips info
    with open(local_json_path, 'r') as f:
        clips_info = json.load(f)
    
    print(f"Found {len(clips_info)} clips for video {video_id}")
    
    # Function to process a single clip
    def process_clip(clip_info):
        video_id = clip_info["video_id"]
        start_time = clip_info["start_time"]
        
        # Get the clip path from the info and convert to GCS URI
        clip_path = clip_info["clip_path"]
        if clip_path.startswith("./"):
            clip_path = clip_path[2:]  # Remove leading "./"
        
        # Construct GCS URI
        clip_uri = f"gs://vlm-testing-vertex-ai/{clip_path}"
        
        # Get description
        description = get_description(vision_model, clip_uri, retry_count=3)
        
        # Return result
        result = {
            "VIDEO": video_id,
            "TIME_START": start_time,
            "DESCRIPTION": description
        }
        
        # Write to CSV immediately to preserve progress
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['VIDEO', 'TIME_START', 'DESCRIPTION'])
            writer.writerow(result)
        
        return result
    
    # Process clips in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_clip, clip) for clip in clips_info]
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"  Progress for {video_id}: {i+1}/{len(clips_info)}")
            except Exception as e:
                print(f"  Error processing clip for {video_id}: {e}")
    
    print(f"Finished processing {len(results)}/{len(clips_info)} clips for video {video_id}")
    return len(results)

def main():
    # Initialize Vertex AI once
    vision_model = init_vertex_ai()
    
    # Prepare output CSV file
    output_csv = "all_video_descriptions.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['VIDEO', 'TIME_START', 'DESCRIPTION'])
        writer.writeheader()
    
    # Find all clips_info.json files
    print("Finding all clips_info.json files...")
    clips_info_files = list_all_clips_info_files()
    
    if not clips_info_files:
        print("No clips_info.json files found!")
        return
    
    print(f"Found {len(clips_info_files)} videos to process:")
    for i, file_uri in enumerate(clips_info_files):
        print(f"  {i+1}. {os.path.basename(file_uri)}")
    
    # Process each video one at a time
    total_clips_processed = 0
    for i, clips_info_uri in enumerate(clips_info_files):
        print(f"\nProcessing video {i+1}/{len(clips_info_files)}: {os.path.basename(clips_info_uri)}")
        clips_processed = process_single_video(clips_info_uri, output_csv, vision_model, max_workers=5)
        total_clips_processed += clips_processed
    
    print(f"\nAll done! Processed {len(clips_info_files)} videos with {total_clips_processed} total clips.")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()