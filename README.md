# VLM Prompt Testing

## Study Objective

In this experiment, we want to explore how different prompts or VLMs used for text inputing will affect the output of predition models. By testing different prompts and different VLMs, we expect to find the prompts and vlms that will maximize the accuracy of robot predictions.

## Tests & Experiments
This research has three different parts:
1. **LLM Comparison**: 
   - Compare the performance of different LLMs (GPT-4o, Gemini 2.0 Flash, Qwen-2.5-vl-72b) on video prediction tasks using various prompts.
   - Evaluate the models' accuracy, precision, recall, F1 score, AUC, MSE, and MAE.
2. **Local Models**:
   - Test local models (Qwen-2.5-vl-7b-Instruct, llava-video, Phi-4) on video prediction tasks.
   - Compare their performance with cloud-based models.
3. **Text Embeddings**:
   - Generate text embeddings for video clips using Gemini 2.0 Flash and prompt 4.1.
   - Analyze the embeddings for potential insights into video content and prediction tasks.

### Tested Prompts & Responses

| No. | **Prompts**                                                                                                                                                                                                                                                                                                                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | "Describe the content of this video in detail. Include information about the environment, objects, people or robots, and their actions. Provide a coherent narrative that explains what is happening in the scene"                                                                                                                                                          |
| 2   | "Analyze this video and describe the following aspects in structured format: 1. Objects Present: List the key objects, 2. Actions: Describe what is happening in the scene, 3. Human or Robot Interaction: If applicable, describe interactions, 4. Emotions or Reactions: Identify any human reactions in the scene. 5. Environment: Describe the setting and background." |
| 3   | "Summarize this video as if explaining it to someone who cannot see it. Focus on key actions, emotions, and interactions. Use clear and vivid language to create a mental picture."                                                                                                                                                                                         |
| 4   | Give me a sentence describing what's going on in the video, keeping only key elements of scenario shown                                                                                                                                                                                                                                                                     |
|4.1|Output a sentence describing what's going on in the video, keeping only key elements of the scenario shown that would allow me to predict the outcome of the situation.|
| 5   | You think this situation ends well or poorly? (Use only one word to answer)                                                                                                                                                                                                                                                                                                 |
|5.1|Given the scenario shown on the video, You think this situation ends well or poorly? (Use only one word to answer)|
| 6   | You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)                                                                                                                                                                                                                                                        |
|6.1|Given the scenario shown on the video, You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)|

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
  - `GPT-4o Prediction (prompt 5.1)`
  - `GPT-4o Prediction (prompt 6.1)`
  - `Qwen Prediction (prompt 5.1)`
  - `Qwen Prediction (prompt 6.1)`
  - `Gemini Prediction (prompt 5.1)`
  - `Gemini Prediction (prompt 6.1)`

### Value Mappings

For binary classification:

- "Poorly" = 1
- "Well" = 0

## Results & Discussion

### Experiment 1(LLM Comparsion) Key Findings

| Modal               | accuracy | precision | recall | f1    | auc   | mse   | mae   |
| ------------------- | -------- | --------- | ------ | ----- | ----- | ----- | ----- |
| GPT-4o (prompt 5.1) | 0.433    | 0.375     | 0.462  | 0.414 | 0.437 | 0.567 | 0.567 |
| GPT-4o (prompt 6.1) | 0.467    | 0.400     | 0.462  | 0.429 | 0.466 | 0.533 | 0.533 |
| Qwen (prompt 5.1)   | 0.500    | 0.450     | 0.692  | 0.545 | 0.523 | 0.500 | 0.500 |
| Qwen (prompt 6.1)   | 0.533    | 0.476     | 0.769  | 0.588 | 0.561 | 0.467 | 0.467 |
| Gemini (prompt 5.1) | 0.700    | 0.625     | 0.769  | 0.690 | 0.708 | 0.300 | 0.300 |
| Gemini (prompt 6.1) | 0.633    | 0.562     | 0.692  | 0.621 | 0.640 | 0.367 | 0.367 |

|                          | accuracy | precision | recall | f1    | auc   | mse   |
| ------------------------ | -------- | --------- | ------ | ----- | ----- | ----- |
| Average Individual Human | 0.621    | 0.575     | 0.599  | 0.579 | 0.619 | 0.379 |

Standard deviation of human participant accuracy: 0.062

Standard deviation of performance across videos:
Avg Individual Human: 0.297
GPT-4o (prompt 5): 0.471
GPT-4o (prompt 6): 0.499
Qwen (prompt 5): 0.496
Qwen (prompt 6): 0.499
Gemini (prompt 5): 0.496
Gemini (prompt 6): 0.500
Gemini (prompt 5.1): 0.458
Gemini (prompt 6.1): 0.482

Standard deviation of model-human agreement across participants:
GPT-4o (prompt 5): 0.082
GPT-4o (prompt 6): 0.084
Qwen (prompt 5): 0.086
Qwen (prompt 6): 0.091
Gemini (prompt 5): 0.075
Gemini (prompt 6): 0.082
Gemini (prompt 5.1): 0.070

accuracy: 0.482
precision: 0.458
recall: 0.458
f1: 0.458

### Experiment 2(Local Models) 

### Experiment 3(Text Embeddings)

#### 1. Model Performance vs True Outcomes

- Gemini (prompt 5.1) achieved the highest accuracy (70.0%) against true outcomes, outperforming the average individual human accuracy (62.1%) by 7.9 percentage points
- **GPT-4o (prompt 5)** achieved the highest accuracy (66.7%) against true outcomes, outperforming the average individual human accuracy (62.1%)
- All other LLM configurations showed lower accuracy than the average individual human, with **GPT-4o (prompt 6)** and **Qwen (prompt 6)** performing worst (46.7%)
- All models with prompt 6 demonstrated perfect or near-perfect recall (100% for GPT-4o, 84.6% for Qwen and Gemini), but at the cost of precision

#### 2. LLM-Human Alignment

- **Gemini (prompt 5.1)** showed the highest alignment with human predictions across both prompts (63.3% for prompt 5, 69.2% for prompt 6)
- **GPT-4o** demonstrated the lowest alignment with human predictions across both prompts (49.0% for prompt 5, 47.1% for prompt 6)
- Models showed consistently higher F1 scores when compared to human predictions than when compared to true outcomes, suggesting humans and models make similar types of errors

#### 3. Per-Video Performance

- **Gemini (prompt 5.1)** outperformed humans on 63.3% of videos, the highest among all models
- **GPT-4o (prompt 6)** and **Qwen (prompt 6)** underperformed on all videos compared to humans, suggesting a consistent bias in their predictions

#### 4. Standard Deviation Analysis

- **Gemini (prompt 5.1)** showed the lowest standard deviation in human agreement (0.070), indicating the most consistent alignment with human judgment
- **Gemini (prompt 5.1)** also showed the lowest standard deviation in performance across videos (0.458), suggesting the most consistent accuracy across different scenarios
- **GPT-4o (prompt 5)** showed the highest standard deviation in human agreement (0.082), indicating the least consistent alignment with human judgment

### Model Comparison

#### GPT-4o

- **GPT-4o (prompt 5)** achieved the second highest accuracy against true outcomes (66.7%) and showed the highest standard deviation in human agreement (0.082)
- **GPT-4o (prompt 6)** achieved the lowest accuracy against true outcomes (46.7%) and showed the highest standard deviation in performance across videos (0.499)

#### Gemini

- Different prompts for Gemini showed significant differences in performance, with **Gemini (prompt 5.1)** achieving the highest accuracy against true outcomes (70.0%) and the lowest standard deviation in human agreement (0.070)
- **Gemini (prompt 6)** the highest standard deviation in performance across videos (0.500)

#### Qwen

- Qwen shows a slimilar pattern to Gemini, with **Qwen (prompt 5)** achieving the same accuracy as **Gemini (prompt 5)** (56.7%) and **Qwen (prompt 6)** achieving the same accuracy as **GPT-4o (prompt 6)** (46.7%)
- Qwen has the largest standard deviation(0.086 and 0.091) of model-human agreement

### Video-Level Analysis

The performance by video reveals interesting patterns:

1. Human accuracy varies substantially across videos, indicating some videos are inherently more difficult to judge
2. LLMs tend toward binary judgments (all correct or all incorrect for a given video)
3. There is only partial overlap between videos that humans find difficult and those that challenge LLMs

The boxplot visualization confirms this pattern, showing much greater variance in human performance compared to the more extreme (all-or-nothing) LLM performance distribution.

### Limitations and Future Directions

1. The binary pattern in LLM predictions suggests they may lack nuance in their judgment compared to humans

2. The high variability in human performance indicates inherent task difficulty or subjectivity that should be further explored

### Conclusion

This analysis demonstrates that state-of-the-art LLMs can match or exceed average individual human performance on video prediction tasks when properly prompted. Gemini with prompt 5.1 achieves the highest accuracy overall, while GPT4o and Qwen show more variability in performance. 

The significant impact of prompt engineering highlights both a challenge and an opportunity—LLM performance can potentially be further improved through careful prompt optimization. The finding that LLMs outperform humans on approximately half of the videos suggests promising avenues for human-AI collaboration in judgment tasks.

## Todo

- [x] GPT-4o finish all prompts testing
- [x] Gemini 2.0 Flash vertex ai api
- [x] Gemini 2.0 Flash finish all prompts testing
- [x] Qwen-2.5-vl API setup
- [x] Qwen-2.5-vl finish all prompts testing
- [x] Extract frames from clips
- [x] Genenrate description for video clips
- [x] Compare VLM predictions vs human predictions
- [x] Compare True Outcome vs human predictions
- [x] Compare VLM predictions vs True Outcome
- [x] Download local model Qwen2.5-vl-7b-Instruct
- [ ] Download more local models

