import torch
from transformers import CLIPTokenizer, CLIPTextModel
import pandas as pd
import numpy as np
import os

class TextEmbeddingProcessor:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        """
        Initialize the TextEmbeddingProcessor with CLIP model from Hugging Face
        Args:
            model_name: CLIP model from Hugging Face (e.g., 'openai/clip-vit-base-patch32')
        """
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()
        print(f"Loaded CLIP model: {model_name}")
        
    def process_csv_file(self, csv_file_path):
        """
        Process a CSV file with the format: VIDEO, TIME_START, DESCRIPTION
        and add CLIP text embeddings for each row
        """
        # Read CSV
        df = pd.read_csv(csv_file_path)
        
        # Check if required columns exist
        required_cols = ['VIDEO', 'TIME_START', 'DESCRIPTION']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")
        
        results = []
        
        # Process each row
        for idx, row in df.iterrows():
            # Clean any NaN values
            text = row['DESCRIPTION']
            if pd.isna(text):
                text = ""  # Empty string for NaN values
            
            # Get CLIP embedding for the text
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77  # CLIP uses 77 as max length
                )
                outputs = self.model(**inputs)
                # Use the pooled output as the text embedding (matches your original implementation)
                embedding = outputs.pooler_output.numpy()[0]
                
                if idx == 0:
                    print('Embedding shape:', embedding.shape)
            
            # Create a dictionary with VIDEO and TIME_START
            result_dict = {
                'VIDEO': row['VIDEO'],
                'TIME_START': row['TIME_START']
            }
            
            # Add each embedding dimension as a separate column
            for i, value in enumerate(embedding):
                result_dict[f'embed_{i}'] = value
            
            results.append(result_dict)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} rows")
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        return result_df
    
    def save_embeddings(self, df, output_path):
        """
        Save the embeddings and metadata to a single CSV file
        """
        # Save all data (VIDEO, TIME_START, and all embedding dimensions) as CSV
        df.to_csv(f"{output_path}clip_text_embeddings.csv", index=False)
        print(f"Saved embeddings to {output_path}clip_text_embeddings.csv")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process CSV file to CLIP text embeddings')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('output_prefix', help='Prefix for output files')
    parser.add_argument('--model', default='openai/clip-vit-base-patch32', 
                        help='CLIP model to use (default: openai/clip-vit-base-patch32)')
    
    args = parser.parse_args()

    # Initialize processor
    processor = TextEmbeddingProcessor(model_name=args.model)

    # Process CSV file
    df = processor.process_csv_file(args.csv_file)

    # Check if the output directory exists, if not create it
    output_dir = args.output_prefix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results
    processor.save_embeddings(df, args.output_prefix)
    print(f"Processed {len(df)} rows")


if __name__ == "__main__":
    main()
    # Example command: python get_text_embeddings_clip.py ../data/all_video_descriptions_cleaned.csv ../data/embeddings/ --model openai/clip-vit-base-patch32