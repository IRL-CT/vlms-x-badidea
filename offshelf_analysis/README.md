# Off-the-Shelf VLM Prompt Testing

## Research Questions

### RQ1 - What is the off-the-shelf predictive power of current VLMs?

- Testing VLMs ability to predict outcome of a scenario shown based only on the context available
- Testing how this predictive power compares with human judgements
- Add. testing: predictive power based only on human reactions or both (reactions + scenario)


## Tested Prompts & Responses


### Scenario Analysis

| No. | **Prompts**                                                                                                                                                                                                                                                                                                                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | "Describe the content of this video in detail. Include information about the environment, objects, people or robots, and their actions. Provide a coherent narrative that explains what is happening in the scene"                                                                                                                                                          |
| 2   | "Analyze this video and describe the following aspects in structured format: 1. Objects Present: List the key objects, 2. Actions: Describe what is happening in the scene, 3. Human or Robot Interaction: If applicable, describe interactions, 4. Emotions or Reactions: Identify any human reactions in the scene. 5. Environment: Describe the setting and background." |
| 3   | "Summarize this video as if explaining it to someone who cannot see it. Focus on key actions, emotions, and interactions. Use clear and vivid language to create a mental picture."                                                                                                                                                                                         |
| 4   | Give me a sentence describing what's going on in the video, keeping only key elements of scenario shown                                                                                                                                                                                                                                                                     |
|4.1|Output a sentence describing what's going on in the video, keeping only key elements of the scenario shown that would allow me to predict the outcome of the situation.|                                                                                                                                      
|5 |Given the scenario shown on the video, you think this situation ends well or poorly? (Use only one word to answer)|
|6|Given the scenario shown on the video, you think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)|


### Reaction analysis

| No. | **Prompts**                                                                                                                                                                                                                                                                                                                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

|1 |Given the reaction shown on the video, you think this situation ends well or poorly? (Use only one word to answer)|
|2 | Given the reaction shown on the video, you think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)|



### Models & Settings

| **Model**        | **Experiment** |
| ---------------- | -------------------- |
| GPT-4o           | LLM comparison      |
| Gemini 2.0 Flash | LLM comparison         |
| Qwen-2.5-vl-72b  | LLM comparison             |
|Qwen|Local Models|
|llava-video|Local Models|
|Phi-4|Local Models|
|Gemini 2.0 Flash|Text Embeddings|
## Dataset

Due to the fact that videos might have different names accoss different data files, this is a reference document for video names.

### Dataset Overview

1. **stimulus_dataset_information.xlxs**(Dataset 1): Maps video filenames to question identifiers
2. **badidea_ground_truth.csv**(Dataset 2): Contains prediction values for each question identifier
3. **analyze-predictions.csv**(Dataset 3): Combined analysis file with predictions from multiple sources

### Key Relationships

The datasets are connected through the following relationship:

```
Final video name (Dataset 1) → Question Mapping on Qualtrics (Dataset 1) → response_video (Dataset 2)
```

Example:

- `6_bad_final_full.mp4` (Final video name) maps to `q_2` (Question Mapping)
- `q_2` appears as a value in the `response_video` column of Dataset 2

### Dataset 1: Video Name Relation Sheet

**Key Columns:**

- Column L: `Final video name` - The filename of each video (e.g., "6_bad_final_full.mp4")
- Column O: `Question Mapping on Qualtrics` - The question identifier (e.g., "q_2")

### Dataset 2: Responses Dataset

**Key Columns:**

- `response_video` - Question identifier matching "Question Mapping on Qualtrics" from Dataset 1
- `class` - Binary values (0 or 1) indicating prediction outcome

### Analysis File: analyze-predictions.csv

**Key Columns:**

- `Video` - Video filenames
- `Question Mapping` - Question identifiers
- `True Outcome` - Binary values (0 or 1)
- Model predictions (all binary 0/1):
  - `GPT-4o Prediction (prompt 5)`
  - `GPT-4o Prediction (prompt 6)`
  - `Qwen Prediction (prompt 5)`
  - `Qwen Prediction (prompt 6)`
  - `Gemini Prediction (prompt 5)`
  - `Gemini Prediction (prompt 6)`

### Value Mappings

For binary classification:

- "Poorly" = 1
- "Well" = 0


## Local Model Installation

Using Ollama on Linux, code based on [this repository](https://github.com/AGRamirezz/BAD-Dog/tree/main/Demo) by collaborator Adolfo Ramirez-Artistizabal.

Prompts: prompt 5 and 6

