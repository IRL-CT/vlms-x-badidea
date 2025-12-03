#!/usr/bin/env python3
"""
Script to predict outcomes from video frame descriptions using various LLMs.
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import json

# Optional imports - will check availability at runtime
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class LLMPredictor:
    """Base class for LLM predictors."""

    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv(self.get_api_key_env_var())

    def get_api_key_env_var(self) -> str:
        """Return the environment variable name for the API key."""
        raise NotImplementedError

    def predict(self, description: str) -> str:
        """Predict outcome from description."""
        raise NotImplementedError


class OpenAIPredictor(LLMPredictor):
    """OpenAI API predictor."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        super().__init__(model_name, api_key)
        self.client = openai.OpenAI(api_key=self.api_key)

    def get_api_key_env_var(self) -> str:
        return "OPENAI_API_KEY"

    def predict(self, description: str) -> str:
        prompt = f"Given the scenario described, do you think this situation ends well or poorly? Use only one word -- well or poorly -- to answer.\n\nScenario: {description}\n\nAnswer:"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )

        return response.choices[0].message.content.strip().lower()


class AnthropicPredictor(LLMPredictor):
    """Anthropic Claude API predictor."""

    def __init__(self, model_name: str = "claude-3-5-haiku-20241022", api_key: str = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        super().__init__(model_name, api_key)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def get_api_key_env_var(self) -> str:
        return "ANTHROPIC_API_KEY"

    def predict(self, description: str) -> str:
        prompt = f"Given the scenario described, do you think this situation ends well or poorly? Use only one word -- well or poorly -- to answer.\n\nScenario: {description}\n\nAnswer:"

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=10,
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text.strip().lower()


class GooglePredictor(LLMPredictor):
    """Google Gemini API predictor."""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        super().__init__(model_name, api_key)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def get_api_key_env_var(self) -> str:
        return "GOOGLE_API_KEY"

    def predict(self, description: str) -> str:
        # Use a more neutral prompt to avoid safety triggers
        #prompt = f"Analyze this scenario and predict the outcome. Respond with exactly one word: 'well' if the situation has a positive resolution, or 'poorly' if it has a negative resolution.\n\nScenario analysis: {description}\n\nPredicted outcome:"
        prompt = f"Given the scenario described, do you think this situation ends well or poorly? Use only one word -- well or poorly -- to answer.\n\nScenario: {description}\n\nAnswer:"


        try:
            # Try with safety settings first
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.0
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            # Check if response was blocked by safety filters
            if not response.candidates:
                return "SAFETY_BLOCKED"
            
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return "SAFETY_BLOCKED"
                
            # Check finish reason
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 3:  # SAFETY
                return "SAFETY_BLOCKED"
            
            return response.text.strip().lower()
            
        except Exception as e:
            error_str = str(e).lower()
           
            print(f"Unexpected error: {e}")
            raise e


def get_predictor(provider: str, model_name: str = None, api_key: str = None) -> LLMPredictor:
    """Factory function to get the appropriate predictor."""
    predictors = {
        'openai': OpenAIPredictor,
        'anthropic': AnthropicPredictor,
        'google': GooglePredictor,
    }

    if provider not in predictors:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(predictors.keys())}")

    predictor_class = predictors[provider]

    if model_name:
        return predictor_class(model_name=model_name, api_key=api_key)
    else:
        return predictor_class(api_key=api_key)


def process_descriptions(
    input_csv: Path,
    output_csv: Path,
    predictor: LLMPredictor,
    delay: float = 4.0,
    skip_existing: bool = True
):
    """Process all descriptions and predict outcomes."""

    # Read existing results if skip_existing is True
    existing_results = {}
    if skip_existing and output_csv.exists():
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['VIDEO'], row['TIME_START'])
                existing_results[key] = row['PREDICTION']
        print(f"Loaded {len(existing_results)} existing predictions")

    # Process descriptions
    results = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} descriptions...")

    for i, row in enumerate(rows):
        video = row['VIDEO']
        time_start = row['TIME_START']
        description = row['DESCRIPTION']

        key = (video, time_start)

        # Skip if already processed
        if key in existing_results:
            prediction = existing_results[key]
            if "ERROR" not in prediction and "SAFETY_BLOCKED" not in prediction and "RATE_LIMITED" not in prediction:
               print(f"[{i+1}/{len(rows)}] Skipping {video} @ {time_start}s (already processed)")
            else:
                try:
                    print(f"[{i+1}/{len(rows)}] Reprocessing {video} @ {time_start}s due to previous error")
                    prediction = predictor.predict(description)
                    print(f"[{i+1}/{len(rows)}] {video} @ {time_start}s: {prediction}")

                    # Rate limiting
                    if i < len(rows) - 1:
                        time.sleep(delay)

                except Exception as e:
                    error_msg = str(e)
                    prediction = "ERROR"
                    print(f"[{i+1}/{len(rows)}] Error processing {video} @ {time_start}s: {e}")
                    
        else:
            try:
                prediction = predictor.predict(description)
                print(f"[{i+1}/{len(rows)}] {video} @ {time_start}s: {prediction}")

                # Rate limiting
                if i < len(rows) - 1:
                    time.sleep(delay)

            except Exception as e:
                error_msg = str(e)
                prediction = "ERROR"
                print(f"[{i+1}/{len(rows)}] Error processing {video} @ {time_start}s: {e}")
                
                
        results.append({
            'VIDEO': video,
            'TIME_START': time_start,
            'DESCRIPTION': description,
            'PREDICTION': prediction
        })

        # Save intermediate results every 10 rows
        if (i + 1) % 10 == 0:
            save_results(results, output_csv)

    # Save final results
    save_results(results, output_csv)
    print(f"\nResults saved to {output_csv}")


def save_results(results: List[Dict[str, Any]], output_csv: Path):
    """Save results to CSV."""
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['VIDEO', 'TIME_START', 'DESCRIPTION', 'PREDICTION']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description='Predict outcomes from video descriptions using LLMs')

    parser.add_argument(
        '--input',
        type=str,
        default='../../data/all_video_descriptions_cleaned.csv',
        help='Input CSV file with descriptions'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for predictions (default: predictions_{provider}_{model}.csv)'
    )

    parser.add_argument(
        '--provider',
        type=str,
        required=True,
        choices=['openai', 'anthropic', 'google'],
        help='LLM provider to use'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Model name (uses default for provider if not specified)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='API key (or set via environment variable)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=4.0,
        help='Delay between API calls in seconds (default: 4.0)'
    )

    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Reprocess all descriptions even if already in output file'
    )

    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    input_csv = script_dir / args.input

    if args.output:
        output_csv = script_dir / args.output
    else:
        model_suffix = args.model.replace('/', '_').replace('-', '_') if args.model else 'default'
        output_csv = script_dir / f'predictions_{args.provider}_{model_suffix}.csv'

    # Create predictor
    try:
        predictor = get_predictor(args.provider, args.model, args.api_key)
        print(f"Using {args.provider} with model: {predictor.model_name}")
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return 1

    # Process descriptions
    try:
        process_descriptions(
            input_csv,
            output_csv,
            predictor,
            delay=args.delay,
            skip_existing=not args.no_skip_existing
        )
    except Exception as e:
        print(f"Error processing descriptions: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

#python predict_outcomes.py --provider google --model gemini-2.0-flash

#nohup python -u predict_outcomes.py --provider google --model gemini-2.0-flash > logs/predict_outcomes_google_gemini_2.0_flash.log 2>&1 &
#nohup python -u predict_outcomes.py --provider google --model gemini-2.5-flash > logs/predict_outcomes_google_gemini_2.5_flash.log 2>&1 &