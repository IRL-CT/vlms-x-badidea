# VLM Prompt Testing
## Study Objective
In this experiment, we want to explore how different prompts or VLMs used for text inputing will affect the output of predition models. By testing different prompts and different VLMs, we expect to find the prompts and vlms that will maximize the accuracy of robot predictions.

## Tests & Experiments

This section documents the prompts and models tested for our current research.

### Tested Prompts & Responses

|No.|**Prompts**                     | 
|----------------|----------------|
|1| "Describe the content of this video in detail. Include information about the environment, objects, people or robots, and their actions. Provide a coherent narrative that explains what is happening in the scene" | 
| 2|"Analyze this video and describe the following aspects in structured format:  1. Objects Present: List the key objects, 2.  Actions: Describe what is happening in the scene, 3. Human or Robot Interaction: If applicable, describe interactions, 4. Emotions or Reactions: Identify any human reactions in the scene. 5. Environment: Describe the setting and background." |
| 3|"Summarize this video as if explaining it to someone who cannot see it. Focus on key actions, emotions, and interactions. Use clear and vivid language to create a mental picture." | 
|4|Give me a sentence describing what's going on in the video, keeping only key elements of scenario shown|
|5|You think this situation ends well or poorly? (Use only one word to answer)|
|6|You think this situation ends well or poorly as if you are a human watching the video? (Use only one word to answer)|

### Models & Settings

| **Model**        | **Other Parameters** |
|----------------|----------------|
| GPT-4o   | OpenAI API     | max_tokens=500 |
| Gemini 2.0 Flash    | top_p=0.9      |
| Qwen-2.5-vl-72b    | top_k=40       |

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

## Results & Discussion
### Key Findings
#### 1. Model Performance vs True Outcomes

- **GPT-4o (prompt 5)** achieved the highest accuracy (66.7%) against true outcomes, outperforming the average individual human accuracy (62.1%)
- All other LLM configurations showed lower accuracy than the average individual human, with **GPT-4o (prompt 6)** and **Qwen (prompt 6)** performing worst (46.7%)
- All models with prompt 6 demonstrated perfect or near-perfect recall (100% for GPT-4o, 84.6% for Qwen and Gemini), but at the cost of precision

#### 2. LLM-Human Alignment

- **Gemini (prompt 5)** showed the highest agreement with individual human predictions (63.2%), followed by **Qwen (prompt 5)** (60.8%)
- **GPT-4o** demonstrated the lowest alignment with human predictions across both prompts (49.0% for prompt 5, 47.1% for prompt 6)
- Models showed consistently higher F1 scores when compared to human predictions than when compared to true outcomes, suggesting humans and models make similar types of errors

#### 3. Per-Video Performance

- LLMs outperformed individual humans on a substantial portion of videos: **Qwen (prompt 5)** and **Gemini (prompt 5)** each outperformed humans on 50% of videos
- The distribution of human accuracy across videos shows much greater variability than LLM accuracy, which tends to be either very high (1.0) or very low (0.0)
- This binary pattern in LLM performance suggests they may be more confident but less nuanced in their judgments

#### 4. Standard Deviation Analysis
- The standard deviation of accuracy among human participants is approximately **0.062** (estimated from human accuracy range), suggesting naturally varying skill levels.
- **GPT-4o (prompt 5)** shows a high standard deviation, reflecting its "all-or-nothing" pattern observed in the box plots
- **Gemini** and **Qwen** models likely show lower standard deviations in their agreement with humans, suggesting they consistently align with human judgment across participants

### Model Comparison

GPT-4o with prompt 5 emerges as the top performer against true outcomes, exceeding average human performance by 4.6 percentage points in accuracy. However, its approach differs from human prediction patterns—it shows the lowest alignment with human judgments among all tested configurations.

In contrast, Gemini and Qwen with prompt 5 align more closely with human predictions while maintaining reasonable accuracy against true outcomes (56.7%). This suggests these models may better capture human-like reasoning patterns, even if they don't achieve the highest overall accuracy.

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

This analysis demonstrates that state-of-the-art LLMs can match or exceed average individual human performance on video prediction tasks when properly prompted. GPT-4o with prompt 5 achieves the highest accuracy overall, while Gemini and Qwen with prompt 5 better align with human judgment patterns. However, models with low standard deviation(Gemini and Qwen) in human agreement are more predictably aligned with human judgment, which may be desirable for applications where human-AI collaboration is important

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
