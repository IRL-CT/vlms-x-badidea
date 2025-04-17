import pandas as pd
import numpy as np

# Load the two datasets
# Replace with your actual file paths
video_mapping_df = pd.read_excel("stimulus_dataset_information.xlsx")
responses_df = pd.read_csv("badidea_ground_truth.csv")

# Step 1: Create a mapping dictionary from "Final video name" to "Question Mapping on Qualtrics"
video_to_question_mapping = dict(zip(
    video_mapping_df["Final video name"], 
    video_mapping_df["Question Mapping on Qualtrics"]
))

# Step 2: Create a mapping from "Question Mapping on Qualtrics" to "response_video"
# Since these have the same values, we can directly map "response_video" to classes
question_to_class = {}
for question_id in video_to_question_mapping.values():
    # Get all rows where response_video matches the question_id
    relevant_responses = responses_df[responses_df["response_video"] == question_id]
    # Calculate the average of the "class" column
    if not relevant_responses.empty:
        avg_class = relevant_responses["class"].mean()
        question_to_class[question_id] = avg_class

# Step 3: Create the final mapping from "Final video name" to average class
video_to_avg_class = {}
for video_name, question_id in video_to_question_mapping.items():
    if question_id in question_to_class:
        video_to_avg_class[video_name] = question_to_class[question_id]

# Step 4: Create a final dataframe for better visualization
result_df = pd.DataFrame({
    "Final video name": list(video_to_avg_class.keys()),
    "Question Mapping": [video_to_question_mapping[video] for video in video_to_avg_class.keys()],
    "Average Class": list(video_to_avg_class.values())
})

# Display the result
print(result_df)

# Optional: Save the result to a CSV file
result_df.to_csv("video_class_averages.csv", index=False)