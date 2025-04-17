import os
from dotenv import load_dotenv
from http import HTTPStatus
import dashscope

# 加载环境变量
load_dotenv()
api_key=os.getenv("DASHSCOPE_API_KEY")
# Set DashScope API Key
dashscope.api_key = api_key  # Explicitly set the API key
messages = [
    {
        "role": "user",
        "content": [
            # Input a video file (replace with your actual video URL)
            {"video": "https://vlm-final-cuts.oss-cn-beijing.aliyuncs.com/60_final.mp4"},
            {"text": "Given the scenario shown on the video, You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)"},
        ]
    }
]

# Call the API
response = dashscope.MultiModalConversation.call(
    model='qwen2.5-vl-72b-instruct',
    messages=messages
)

# Check response
if response.status_code == HTTPStatus.OK:
    print(response)
else:
    print(f"Error Code: {response.code}")
    print(f"Error Message: {response.message}")