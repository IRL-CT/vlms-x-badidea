import os
from openai import OpenAI
from dotenv import load_dotenv
from http import HTTPStatus
import dashscope

# 加载环境变量
load_dotenv()
api_key=os.getenv("DASHSCOPE_API_KEY")
# Set DashScope API Key
dashscope.api_key = api_key  # Explicitly set the API key
# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key=api_key,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
messages = [
    {
        "role": "user",
        "content": [
            # Input a video file (replace with your actual video URL)
            {"video": "https://vlm-final-cuts.oss-cn-beijing.aliyuncs.com/60_final.mp4"},
            {"text": "You think this situation ends well or poorly? (Use only one word to answer)"},
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