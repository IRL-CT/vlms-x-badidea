import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import os

class TextEmbeddingProcessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
    
    def process_csv_file(self, csv_file_path):
        """
        Process a CSV file with the format: VIDEO, TIME_START, DESCRIPTION
        and add BERT embeddings for each row
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
            # Get BERT embedding for the text
            with torch.no_grad():
                inputs = self.tokenizer(row['DESCRIPTION'], 
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=512)
                outputs = self.model(**inputs)
                # Use [CLS] token embedding as sentence embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
                
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
        df.to_csv(f"{output_path}text_embeddings.csv", index=False)
        print(f"Saved embeddings to {output_path}text_embeddings.csv")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process CSV file to BERT embeddings')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('output_prefix', help='Prefix for output files')
    parser.add_argument('--model', default='bert-base-uncased', 
                        help='BERT model to use (default: bert-base-uncased)')
    
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
    # Example command: python get_text_embeddings.py ../data/all_video_descriptions_cleaned.csv ../data/embeddings/ --model bert-base-uncased