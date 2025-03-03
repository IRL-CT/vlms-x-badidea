import vertexai
import csv
from vertexai.generative_models import GenerativeModel, Part

PROJECT_ID = "gen-lang-client-0087070594"

vertexai.init(project=PROJECT_ID, location="us-central1")

vision_model = GenerativeModel("gemini-2.0-flash")

image_uris = [f"gs://vlm-testing-vertex-ai/output_frames/59_final/frame_{i:04d}.png" for i in range(0, 17)]  # Adjust range as needed

prompt = "Summarize this video as if explaining it to someone who cannot see it. Focus on key actions, emotions, and interactions. Use clear and vivid language to create a mental picture."

csv_filename = "output.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Image Name", "Generated Text"])  # 写入表头

    for index, image_uri in enumerate(image_uris, start=1):
        image_name = f"frame_{index-1:04d}.png"  # 提取文件名
        print(f"\n Processing Image {index-1}: {image_uri}")

        try:
            response = vision_model.generate_content([
                Part.from_uri(image_uri, mime_type="image/png"),
                Part.from_text(prompt)
            ])

            generated_text = response.text.strip() if response.text else "No response"
            writer.writerow([image_name, generated_text])

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            writer.writerow([image_name, "Error"])

print(f"\n✅ All results saved to {csv_filename}")