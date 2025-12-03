# VLM-LLM Outcome Prediction

This folder contains scripts to predict outcomes from video frame descriptions using various Large Language Models (LLMs).

## Setup

### Install Dependencies

```bash
# For OpenAI
pip install openai

# For Anthropic (Claude)
pip install anthropic

# For Google (Gemini)
pip install google-generativeai

# Install all
pip install openai anthropic google-generativeai
```

### Set API Keys

Set your API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"

# Google
export GOOGLE_API_KEY="your-api-key-here"
```

Or pass it directly using the `--api-key` argument.

## Usage

### Basic Usage

```bash
# Using OpenAI (default: gpt-4o-mini)
python predict_outcomes.py --provider openai

# Using Anthropic Claude (default: claude-3-5-haiku-20241022)
python predict_outcomes.py --provider anthropic

# Using Google Gemini (default: gemini-1.5-flash)
python predict_outcomes.py --provider google
```

### Specify Custom Model

```bash
# OpenAI with GPT-4
python predict_outcomes.py --provider openai --model gpt-4o

# Anthropic with Claude Sonnet
python predict_outcomes.py --provider anthropic --model claude-3-5-sonnet-20241022

# Google with Gemini Pro
python predict_outcomes.py --provider google --model gemini-1.5-pro
```

### Advanced Options

```bash
# Custom input/output files
python predict_outcomes.py \
    --provider openai \
    --input ../../data/all_video_descriptions_cleaned.csv \
    --output my_predictions.csv

# Adjust rate limiting (delay between API calls)
python predict_outcomes.py --provider openai --delay 1.0

# Reprocess all (don't skip existing predictions)
python predict_outcomes.py --provider openai --no-skip-existing

# Pass API key directly
python predict_outcomes.py --provider openai --api-key "your-key-here"
```

## Output Format

The script generates a CSV file with the following columns:
- `VIDEO`: Video identifier
- `TIME_START`: Timestamp of the frame
- `DESCRIPTION`: Frame description
- `PREDICTION`: LLM's prediction (well/poorly)

Default output filename: `predictions_{provider}_{model}.csv`

## Features

- **Multiple LLM Providers**: Support for OpenAI, Anthropic, and Google
- **Resume Capability**: Automatically skips already-processed descriptions
- **Intermediate Saves**: Saves progress every 10 rows
- **Rate Limiting**: Configurable delay between API calls
- **Error Handling**: Continues processing on errors, marks failed predictions as "ERROR"

## Examples

### Running Multiple Models in Parallel

```bash
# Terminal 1
python predict_outcomes.py --provider openai --model gpt-4o-mini

# Terminal 2
python predict_outcomes.py --provider anthropic --model claude-3-5-haiku-20241022

# Terminal 3
python predict_outcomes.py --provider google --model gemini-1.5-flash
```

### Analyzing Predictions

```python
import pandas as pd

# Load predictions
df = pd.read_csv('predictions_openai_gpt_4o_mini.csv')

# Count predictions
print(df['PREDICTION'].value_counts())

# Compare across models
openai_df = pd.read_csv('predictions_openai_gpt_4o_mini.csv')
claude_df = pd.read_csv('predictions_anthropic_claude_3_5_haiku_20241022.csv')

# Merge and compare
merged = openai_df.merge(claude_df, on=['VIDEO', 'TIME_START'], suffixes=('_openai', '_claude'))
agreement = (merged['PREDICTION_openai'] == merged['PREDICTION_claude']).mean()
print(f"Agreement rate: {agreement:.2%}")
```
