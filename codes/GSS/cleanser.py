import pandas as pd
import numpy as np
import csv
import os
from DIFY_none_RAG import COLUMNS
def extract_and_sample_csv(input_csv_path, output_csv_path, columns_to_extract, sample_size):
    """
    Extract specified columns from CSV file, randomly sample specified number of rows from
    rows that have complete values in all specified columns, and save to new CSV.
    
    Args:
        input_csv_path (str): Path to input CSV file
        output_csv_path (str): Path to output CSV file
        columns_to_extract (list): List of column names to extract
        sample_size (int|None): Number of rows to randomly sample; if None, take all complete rows
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv_path)
        print(f"Original CSV shape: {df.shape}")
        #print(f"Available columns: {list(df.columns)}")
        
        # Check if all required columns exist
        missing_columns = [col for col in columns_to_extract if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")
        
        # Extract only the specified columns
        df_extracted = df[columns_to_extract]
        print(f"Extracted columns: {columns_to_extract}")
        print(f"Extracted data shape: {df_extracted.shape}")
        
        # Keep only rows where all specified columns are non-missing
        df_complete = df_extracted.dropna(how='any', subset=columns_to_extract)
        # Additionally, filter out empty-string-like values if present
        def is_complete_row(row):
            return all(str(v).strip() != '' for v in row.values)
        df_complete = df_complete[df_complete.apply(is_complete_row, axis=1)]
        print(f"Complete rows available: {len(df_complete)}")
        
        # Determine sample size
        if sample_size is None:
            sample_size_use = len(df_complete)
        else:
            sample_size_use = sample_size
        
        # Randomly sample up to the requested number of rows
        if len(df_complete) >= sample_size_use:
            df_sampled = df_complete.sample(n=sample_size_use, random_state=42)
        else:
            print(f"Warning: Only {len(df_complete)} complete rows available; requested {sample_size_use}. Returning all complete rows.")
            df_sampled = df_complete.copy()
        print(f"Sampled data shape: {df_sampled.shape}")
        
        # Save to new CSV file
        df_sampled.to_csv(output_csv_path, index=False)
        print(f"Successfully saved {len(df_sampled)} rows to {output_csv_path}")
        
        # Display some statistics
        print("\nData Statistics (sampled):")
        for col in columns_to_extract:
            print(f"{col}: {df_sampled[col].value_counts(dropna=False).to_dict()}")
            
    except Exception as e:
        print(f"Error processing CSV: {e}")

def main():
    # Configuration
    input_csv_path = "/home/cyyuan/FAS_PRO/data/gss/GSS2024.csv"  # Path to your input CSV
    output_csv_path = "/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_edc.csv"    # Path for output CSV
    columns_to_extract = COLUMNS  # Columns to extract
    sample_size = None  # Set to None to take all rows with complete values
    
    # Check if input file exists
    if not os.path.exists(input_csv_path):
        print(f"Input file not found: {input_csv_path}")
        print("Please make sure the CSV file exists or update the path.")
        return
    
    # Extract and sample data
    extract_and_sample_csv(input_csv_path, output_csv_path, columns_to_extract, sample_size)

if __name__ == "__main__":
    main()