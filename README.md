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

1. **stimulus_dataset_information.xlxs**: Maps video filenames to question identifiers
2. **badidea_ground_truth.csv**: Contains prediction values for each question identifier
3. **analyze-predictions.csv**: Combined analysis file with predictions from multiple sources

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
- `Average Class of Human Prediction` - Continuous values between 0-1
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

Based on the comprehensive analysis of prediction performance across human judgments and multiple VLM models (GPT-4o, Qwen, and Gemini), some significant patterns can be found:

#### 1. Overall Performance Ranking

- **Human predictions** demonstrated the highest accuracy (73.3%) and F1 score (0.692) against true outcomes, outperforming all VLM models
- Among VLMs, **GPT-4o with prompt 5** achieved the highest accuracy (66.7%) and AUC (0.652)
- For most metrics, the performance ranking follows: Human > GPT-4o (prompt 5) > Qwen/Gemini (prompt 5) > Gemini (prompt 6) > GPT-4o/Qwen (prompt 6)

#### 2. Prompt Effectiveness

- **Prompt 5 consistently outperformed prompt 6** across all three VLM models for accuracy and precision
- Interestingly, **prompt 6 yielded higher recall** across all models, suggesting it may bias models toward positive predictions (class 1)
- The performance gap between prompts was most pronounced for GPT-4o (20 percentage points in accuracy)

#### 3. Model Characteristics

- **GPT-4o (prompt 5)** demonstrated the best balance between precision (0.636) and recall (0.538)
- **Qwen and Gemini** showed identical performance patterns with prompt 5 (accuracy: 56.7%, F1: 0.629)
- All models with **prompt 6** exhibited perfect or near-perfect recall (1.0 for GPT-4o and Qwen, 0.846 for Gemini) but at the cost of precision (~0.45)

#### 4. Agreement Patterns

- The agreement matrix reveals that **Qwen and Gemini models strongly agree with each other** (86.7-93.3% agreement)
- **GPT-4o (prompt 5)** shows relatively low agreement with other models (33.3-50.0%)
- **Human predictions** agree most strongly with the true outcome (73.3%) compared to any VLM

### Detailed Analysis

#### Human vs. VLM Performance

Human predictions significantly outperformed all VLM configurations. With an accuracy of 73.3% and balanced precision and recall (both 0.692), humans demonstrated superior ability to discern the correct outcomes. This may suggest that the task involves nuanced judgment that humans are better equipped to handle.

The closest VLM performance came from GPT-4o (prompt 5) at 66.7% accuracy, which is still notably lower than human performance. The substantial gap suggests that current VLMs, while impressive, still fall short of human-level discernment for this specific prediction task.

#### Prompt Engineering Implications

The dramatic difference in performance between prompts 5 and 6 highlights the critical importance of prompt engineering. For GPT-4o, the choice of prompt created a 20 percentage point difference in accuracy (66.7% vs. 46.7%).

The consistent pattern across all models where prompt 6 yields higher recall but lower precision suggests that this prompt biases the models toward predicting class 1 ("Poorly"). This observation is further supported by the confusion matrices showing many false positives for prompt 6 configurations.

#### Model Agreement and Clustering

The agreement matrix reveals three distinct clusters:
1. Human predictions and true outcomes (73.3% agreement)
2. GPT-4o (prompt 5) as an outlier with unique prediction patterns
3. All remaining VLM configurations showing high inter-model agreement

This clustering suggests that Qwen and Gemini may employ similar reasoning patterns or have been trained on similar data, while GPT-4o's approach differs significantly. Human predictions appear to capture signals that none of the VLMs consistently detect.

## Conclusion

This analysis shows that while VLMs can achieve reasonable performance on this prediction task, they still lag behind human judgment. GPT-4o with appropriate prompting (prompt 5) comes closest to human-level performance, but the gap remains substantial. The significant impact of prompt engineering highlights both a challenge and an opportunity: VLM performance can potentially be further improved through prompt optimization.

The similarity in performance patterns among certain models (Qwen and Gemini) contrasted with the distinct behavior of GPT-4o suggests that different architectural or training approaches may be better suited for different aspects of the task. Future work could explore ensemble methods that leverage the strengths of each model.

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
