"""
Vision Language Model Demo for Webcam Scene Analysis

This script uses a webcam to capture live video frames and sends them to a Vision-Language Model
(VLM) for real-time scene analysis and description. The system uses Ollama to access
open-source multimodal models like Llama 3.2 Vision for completely local inference.
"""

import cv2
import time
import base64
import requests
import json
import argparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class VLMDemo:
    def __init__(self, 
                 model="llama3-2-vision", 
                 prompt="Given the reaction shown on the video, you think this situation ends well or poorly? (Use only one word to answer)",
                 ollama_url="http://localhost:11434",
                 camera_id=0,
                 fps_target=1,
                 display_width=1280,
                 display_height=720):
        
        self.model = model
        self.base_prompt = prompt
        self.ollama_url = ollama_url
        self.camera_id = camera_id
        self.fps_target = fps_target
        self.interval = 1.0 / fps_target
        self.display_width = display_width
        self.display_height = display_height
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open webcam.")
            
        # Set up font for displaying text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Last analysis time and result
        self.last_analysis_time = 0
        self.last_result = "Starting analysis..."
        
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 for API request"""
        # Convert BGR to RGB (PIL uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Resize image to reduce payload size
        pil_img = pil_img.resize((640, 360))
        
        # Save image to BytesIO object
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        
        # Get base64 string
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    
    def analyze_image(self, frame):
        """Send image to Ollama for analysis"""
        try:
            # Convert frame to base64
            base64_image = self.frame_to_base64(frame)
            
            # Build the prompt with the current system time
            current_time = time.strftime("%H:%M:%S")
            prompt = f"Current time: {current_time}. {self.base_prompt}"
            
            # Prepare Ollama API request
            api_url = f"{self.ollama_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt,
                        "images": [base64_image]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096
                }
            }
            
            # Make the API request
            response = requests.post(api_url, json=payload)
            response_data = response.json()
            
            # Extract the response
            if 'message' in response_data:
                description = response_data['message']['content']
                return description
            else:
                return "Error: Could not process the image."
        
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def overlay_text(self, frame, text):
        """Overlay text on frame with proper wrapping and formatting"""
        # Create a dark overlay for better text visibility
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Add semi-transparent black overlay at bottom
        cv2.rectangle(overlay, (0, h - 230), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Wrap and render text
        font_scale = 0.7
        thickness = 1
        color = (255, 255, 255)
        max_width = w - 20
        
        # Wrap text to multiple lines
        words = text.split(' ')
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = cv2.getTextSize(word + ' ', self.font, font_scale, thickness)[0][0]
            if current_width + word_width > max_width:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                current_line.append(word)
                current_width += word_width
                
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw lines
        y_position = h - 210
        line_height = 30
        
        for i, line in enumerate(lines[:6]):  # Limit to 6 lines
            cv2.putText(frame, line, (10, y_position + i * line_height), 
                        self.font, font_scale, color, thickness)
            
        if len(lines) > 6:
            cv2.putText(frame, "...", (10, y_position + 6 * line_height), 
                        self.font, font_scale, color, thickness)
            
        # Add status line at top
        status = f"Model: {self.model} | Press 'Q' to quit | 'A' to analyze now"
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 30), self.font, 0.7, (255, 255, 255), 1)
    
    def run(self):
        """Main loop for capturing and analyzing frames"""
        print(f"Starting webcam analysis with {self.model}...")
        print(f"Press 'Q' to quit or 'A' to force analysis of current frame")
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break
                
            # Calculate time elapsed since last analysis
            current_time = time.time()
            time_elapsed = current_time - self.last_analysis_time
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Check if it's time to analyze or if user pressed 'A'
            key = cv2.waitKey(1) & 0xFF
            if (time_elapsed >= self.interval) or (key == ord('a')):
                print("Analyzing current frame...")
                self.last_analysis_time = current_time
                
                # Run analysis in the main thread
                self.last_result = self.analyze_image(frame)
                print(f"Analysis result: {self.last_result}")
            
            # Overlay the most recent result
            self.overlay_text(display_frame, self.last_result)
            
            # Display frame
            cv2.imshow('VLM Demo - Scene Analysis', display_frame)
            
            # Check for exit
            if key == ord('q'):
                break
                
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        
def main():
    parser = argparse.ArgumentParser(description='Webcam Scene Analysis using Vision-Language Models')
    parser.add_argument('--model', type=str, default='llama3-2-vision', 
                        help='Ollama model to use (default: llama3-2-vision)')
    parser.add_argument('--prompt', type=str, 
                        default='Describe what you see in detail, including people, objects, actions, and the setting.',
                        help='Prompt for the vision model')
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera device ID (default: 0)')
    parser.add_argument('--fps', type=float, default=0.2,
                        help='Target analysis frames per second (default: 0.2 - once every 5 seconds)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='Ollama API URL (default: http://localhost:11434)')
    
    args = parser.parse_args()
    
    demo = VLMDemo(
        model=args.model,
        prompt=args.prompt,
        ollama_url=args.ollama_url,
        camera_id=args.camera,
        fps_target=args.fps
    )
    
    demo.run()

if __name__ == "__main__":
    main()
