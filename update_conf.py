#!/usr/bin/env python3
"""
Script to update CSV files by:
1. Renaming 'prob' column to 'conf1'
2. Adding 'manual_conf' column that equals 'conf1' if 'manual_label' == 'correct_label', otherwise 1
"""

import pandas as pd
import os
import glob
from pathlib import Path
import sys

def update_csv_file(file_path):
    """
    Update a single CSV file with the required column changes.
    
    Args:
        file_path (str): Path to the CSV file to update
    """
    print(f"Processing: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
        
    # Rename 'prob' to 'conf1'
    df = df.rename(columns={'prob': 'confs1'})
    
    # Add 'manual_conf' column
    # If manual_label == correct_label, use conf1 value, otherwise use 1
    df['manual_conf'] = df.apply(
        lambda row: row['confs1'] if row['manual_label'] == row['correct_label'] else 0.9999, 
        axis=1
    )
    
    # Save the updated file
    df.to_csv(file_path, index=False)
    print(f"  Updated: {file_path}")

    return df

def main(csv_dir):
    """
    Main function to process all CSV files in the specified directory.
    """
    
    # Check if directory exists
    if not os.path.exists(csv_dir):
        print(f"Error: Directory {csv_dir} does not exist!")
        return
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file_path in csv_files:
        print(f"  - {os.path.basename(file_path)}")
    
    print("\nStarting processing...")
    
    # Process each CSV file
    for file_path in csv_files:
        try:
            update_csv_file(file_path)
        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
        # Define the directory path
    if len(sys.argv) < 2:
        print("Usage: python update_conf.py <csv_dir>")
        sys.exit(1)
    csv_dir = sys.argv[1]
    main(csv_dir=csv_dir) 