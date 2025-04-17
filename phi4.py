import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"
device = "cuda:0"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device,  torch_dtype=torch.float16)
