import vertexai

from vertexai.generative_models import GenerativeModel, Part

PROJECT_ID = "gen-lang-client-0087070594"

vertexai.init(project=PROJECT_ID, location="us-central1")

vision_model = GenerativeModel("gemini-2.0-flash")
# Max 10 videos per prompt
video_uri_first_ten = [
    "gs://vlm-testing-vertex-ai/final_cut_videos/6_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/7_final.mp4",
    # "gs://vlm-testing-vertex-ai/final_cut_videos/9_final.mp4",
    # "gs://vlm-testing-vertex-ai/final_cut_videos/11_final.mp4",
    # "gs://vlm-testing-vertex-ai/final_cut_videos/12_final.mp4",
    # "gs://vlm-testing-vertex-ai/final_cut_videos/14_final.mp4",
    # "gs://vlm-testing-vertex-ai/final_cut_videos/15_final.mp4",
    # "gs://vlm-testing-vertex-ai/final_cut_videos/19_final.mp4",
    # "gs://vlm-testing-vertex-ai/final_cut_videos/21_final.mp4",
    # "gs://vlm-testing-vertex-ai/final_cut_videos/22_final.mp4",
]
video_uri_second_ten = [
    "gs://vlm-testing-vertex-ai/final_cut_videos/24_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/29_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/30_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/31_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/33_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/34_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/40_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/41_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/44_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/46_final.mp4",
]
video_uri_third_ten = [
    "gs://vlm-testing-vertex-ai/final_cut_videos/48_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/49_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/50_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/52_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/53_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/54_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/55_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/57_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/59_final.mp4",
    "gs://vlm-testing-vertex-ai/final_cut_videos/60_final.mp4",
]

prompts={
    # "Prompt 1:":"Describe the content of this video in detail. Include information about the environment, objects, people or robots, and their actions. Provide a coherent narrative that explains what is happening in the scene",
    # "Prompt 2:":"Summarize this video as if explaining it to someone who cannot see it. Focus on key actions, emotions, and interactions. Use clear and vivid language to create a mental picture.",
    # "Prompt 3:":"Analyze the video using an action recognition approach. Identify 1. Key movements, 2. Object manipulations, 3. Gestures or postures, 4. Any interactions between humans and robots. Provide a precise technical breakdown of movements",
    # "Promp 4": "Output a sentence describing what's going on in the video, keeping only key elements of scenario shown",
    # "Prediction": "You think this situation ends well or poorly? (Use only one word to answer)",
    # "Prompt 6": "You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)",
    # "Prompt 5.1": "Given the scenario shown on the video, You think this situation ends well or poorly? (Use only one word to answer)",
    # "Prompt 6.1":"Given the scenario shown on the video, You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)",
    "Prompt 4.1": "Output a sentence describing what's going on in the video, keeping only key elements of the scenario shown that would allow me to predict the outcome of the situation.",
}   

for video_uri in video_uri_first_ten:
    print(f"\n Processing video: {video_uri}")

    for prompt_name, prompt_text in prompts.items():
        print(f"\n Running prompt: {prompt_name}")

        response = vision_model.generate_content([
            Part.from_uri(video_uri, mime_type="video/mp4"),
            Part.from_text(prompt_text)
        ])

        print(f"\n Response ({prompt_name}):\n{response.text}")