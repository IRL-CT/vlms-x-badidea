# Local Vision-Language Model (VLM) Scenario Analysis

This directory contains scripts and analysis for evaluating local Vision-Language Models on scenario outcome prediction tasks. The system tests multiple VLMs to predict whether situations shown in videos will end "well" or "poorly" based on visual information.

## Overview

This analysis evaluates various open-source VLMs running locally via Ollama, comparing their ability to predict outcomes from video scenarios. The models analyze video frames and make binary predictions about scenario outcomes, which are then compared against ground truth labels to compute performance metrics.

## Directory Structure

```
local/
├── vlm_reactions.py              # Frame-by-frame video analysis script
├── vlm_reactions_video.py        # Full video analysis script (multiple frames at once)
├── results/                      # Raw model predictions per video
│   ├── results_gemma3.csv
│   ├── results_gemma3_27b.csv
│   ├── results_llama32vision.csv
│   ├── results_llama4.csv
│   ├── results_llava.csv
│   ├── results_llavallama3.csv
│   ├── results_qwen25.csv
│   └── test_results.csv
├── analysis/                     # Processed results and analysis
│   ├── processing_results.ipynb  # Main analysis notebook
│   ├── all_predictions.csv       # All frame-level predictions
│   ├── all_predictions_grouped.csv  # Aggregated predictions per video
│   ├── all_predictions_grouped_with_true_outcome.csv  # With ground truth
│   └── prediction_performance.csv  # Model performance metrics
└── logs/                         # Execution logs from model runs
```

## Files Description

### Python Scripts

#### `vlm_reactions.py`
Frame-by-frame video analyzer that:
- Samples frames from videos at a specified rate (default: every 15th frame)
- Sends each frame individually to a local VLM via Ollama API
- Collects predictions for each frame
- Saves results to CSV with columns: `video_name`, `frame`, `outcome_prediction`

**Key Parameters:**
- `--model`: VLM model name (e.g., llama3.2-vision, llava, qwen2.5vl)
- `--video-folder`: Path to input videos
- `--output-csv`: Output file path
- `--frame-sample-rate`: Sample every Nth frame (default: 30)
- `--ollama-url`: Ollama API endpoint (default: http://localhost:11434)

**Usage Example:**
```bash
python vlm_reactions.py \
  --model llava \
  --video-folder '../../../data/final_cut_videos/' \
  --output-csv './results/results_llava.csv' \
  --frame-sample-rate 15
```

#### `vlm_reactions_video.py`
Full video analyzer that:
- Extracts multiple frames from each video (up to 45 frames)
- Sends all frames together as a batch to the VLM
- Gets a single prediction per video based on all frames
- More context-aware but requires models with larger context windows

### Analysis Notebook

#### `processing_results.ipynb`
The main analysis notebook that processes raw VLM predictions and computes performance metrics.

**What it does:**

1. **Data Loading & Aggregation:**
   - Loads all `results_*.csv` files from the `results/` directory
   - Removes error responses and NaN values
   - Groups frame-level predictions by video using mode (most common prediction)
   - Creates unified dataframe with all model predictions

2. **Prediction Distribution Analysis:**
   - Counts "Well" vs "Poorly" predictions per model
   - Shows prediction bias across different models

3. **Ground Truth Matching:**
   - Loads ground truth labels from `../../../dataset_scenarios/analyze_predictions.csv`
   - Maps video names to true outcomes (0 = Well, 1 = Poorly)
   - Converts text predictions to numeric labels

4. **Performance Evaluation:**
   - Computes classification metrics per model:
     - **Accuracy**: Overall correctness
     - **Precision**: Positive predictive value (correctly identified "poorly" outcomes)
     - **Recall**: Sensitivity (proportion of actual "poorly" outcomes identified)
     - **F1 Score**: Harmonic mean of precision and recall
   - Saves metrics to `prediction_performance.csv`

**Output Files Generated:**
- `all_predictions.csv`: All individual frame predictions (3,534 rows after cleaning)
- `all_predictions_grouped.csv`: Mode predictions per video (180 videos × 6 models)
- `all_predictions_grouped_with_true_outcome.csv`: Predictions with ground truth labels
- `prediction_performance.csv`: Performance metrics for each model

## Results Summary

### Model Performance

Based on `prediction_performance.csv`, the models achieved the following metrics:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **llavallama3** | **0.533** | 0.400 | 0.154 | 0.222 |
| **qwen25** | **0.500** | 0.429 | 0.462 | 0.444 |
| gemma3 | 0.433 | 0.433 | 1.000 | 0.605 |
| gemma3_27b | 0.433 | 0.433 | 1.000 | 0.605 |
| llama32vision | 0.433 | 0.433 | 1.000 | 0.605 |
| llava | 0.333 | 0.360 | 0.692 | 0.474 |

### Key Findings

1. **Best Overall Accuracy**: `llavallama3` (53.3%) and `qwen25` (50.0%) achieved the highest accuracy
2. **Prediction Bias**:
   - `gemma3`, `gemma3_27b`, and `llama32vision` predicted "Poorly" for all 30 videos (100% recall, but lower accuracy)
   - `llava` showed heavy bias toward "Poorly" (25 out of 30 videos)
   - `llavallama3` showed opposite bias toward "Well" (25 out of 30 videos)
   - `qwen25` was most balanced (14 "Poorly" vs 16 "Well")

3. **Best Balanced Performance**: `qwen25` showed the most balanced predictions with reasonable accuracy (50%)

4. **Dataset Composition**: The test set contains 30 unique videos per model after filtering errors

## Running the Analysis

### Prerequisites
```bash
conda activate ollama
# Pull required models
ollama pull llama3.2-vision
ollama pull llava
ollama pull qwen2.5vl
# ... other models
```

### Generate Predictions
```bash
# Run for a specific model
nohup python vlm_reactions.py \
  --model llava \
  --video-folder '../../../data/final_cut_videos/' \
  --output-csv './results/results_llava.csv' \
  --frame-sample-rate 15 > ./logs/vlm_output_llava.log 2>&1 &
```

### Process Results
Open and run `analysis/processing_results.ipynb` in Jupyter to:
1. Aggregate all model predictions
2. Compare against ground truth
3. Generate performance metrics
4. Export processed CSVs

## Notes

- The analysis filters out error responses (rows containing "Error" in predictions)
- Frame-level predictions are aggregated using mode (most common prediction) per video
- Original dataset: 4,123 frame predictions → 3,534 after cleaning
- Final aggregated dataset: 180 video-level predictions (30 videos × 6 models)
- Temperature setting: 0.2 (relatively deterministic responses)
- Context window: 4096 tokens
