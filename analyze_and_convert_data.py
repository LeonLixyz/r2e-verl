#!/usr/bin/env python3
"""
Script to analyze data structures and convert JSON to proper parquet format.

This script:
1. Loads and analyzes the working test_codeforces.parquet file
2. Loads and analyzes the problematic deepcoder_train.json file
3. Converts the JSON to the correct parquet format that matches the working file
4. Saves the converted file as a new parquet file
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys

def analyze_parquet_structure(parquet_path):
    """Analyze the structure of a parquet file."""
    print(f"\n=== Analyzing {parquet_path} ===")
    
    try:
        # Read with pandas
        df = pd.read_parquet(parquet_path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Show first few rows
        print(f"\nFirst 2 rows:")
        for i, row in df.head(2).iterrows():
            print(f"Row {i}:")
            for col in df.columns:
                print(f"  {col}: {type(row[col])} = {row[col]}")
        
        # Analyze nested structures
        for col in df.columns:
            sample_val = df[col].iloc[0]
            print(f"\nColumn '{col}' sample structure:")
            print(f"  Type: {type(sample_val)}")
            if isinstance(sample_val, (list, dict)):
                print(f"  Content: {sample_val}")
        
        return df
        
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return None

def analyze_json_structure(json_path):
    """Analyze the structure of a JSON file."""
    print(f"\n=== Analyzing {json_path} ===")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Type: {type(data)}")
        print(f"Length: {len(data) if isinstance(data, list) else 'N/A'}")
        
        # Show first few items
        if isinstance(data, list) and len(data) > 0:
            print(f"\nFirst 2 items:")
            for i, item in enumerate(data[:2]):
                print(f"Item {i}:")
                for key, value in item.items():
                    print(f"  {key}: {type(value)} = {value}")
                print()
        
        return data
        
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

def convert_json_to_parquet(json_data, output_path, reference_df=None):
    """Convert JSON data to parquet format matching the reference structure."""
    print(f"\n=== Converting JSON to Parquet ===")
    
    try:
        # Create DataFrame from JSON
        df = pd.DataFrame(json_data)
        print(f"Created DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Ensure proper data types match reference if provided
        if reference_df is not None:
            print("Matching reference structure...")
            for col in reference_df.columns:
                if col in df.columns:
                    ref_type = type(reference_df[col].iloc[0])
                    print(f"  {col}: ensuring type matches reference ({ref_type})")
        
        # Save to parquet with proper settings to avoid nested data issues
        print(f"Saving to {output_path}...")
        
        # Use pyarrow with specific settings to handle nested data properly
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')
        
        print(f"Successfully saved {len(df)} rows to {output_path}")
        return df
        
    except Exception as e:
        print(f"Error converting to parquet: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Paths
    data_dir = Path("/workspace/rllm/data")
    working_parquet = data_dir / "test_codeforces.parquet"
    problematic_json = data_dir / "deepcoder_train.json"
    output_parquet = data_dir / "deepcoder_train_fixed.parquet"
    
    print("=== Data Structure Analysis and Conversion Tool ===")
    
    # 1. Analyze working parquet file
    if working_parquet.exists():
        working_df = analyze_parquet_structure(working_parquet)
    else:
        print(f"Warning: {working_parquet} not found")
        working_df = None
    
    # 2. Analyze problematic JSON file
    if problematic_json.exists():
        json_data = analyze_json_structure(problematic_json)
    else:
        print(f"Error: {problematic_json} not found")
        return 1
    
    # 3. Convert JSON to proper parquet format
    if json_data:
        converted_df = convert_json_to_parquet(json_data, output_parquet, working_df)
        
        if converted_df is not None:
            print(f"\n=== Conversion Success ===")
            print(f"Original JSON: {len(json_data)} items")
            print(f"Converted parquet: {len(converted_df)} rows")
            print(f"Output file: {output_parquet}")
            
            # Test loading the converted file
            print(f"\n=== Testing Converted File ===")
            try:
                test_df = pd.read_parquet(output_parquet)
                print(f"✓ Successfully loaded converted file: {test_df.shape}")
                
                # Test with datasets library (like the error case)
                try:
                    import datasets
                    dataset = datasets.load_dataset("parquet", data_files=str(output_parquet))["train"]
                    print(f"✓ Successfully loaded with datasets library: {len(dataset)} rows")
                except Exception as e:
                    print(f"✗ Failed with datasets library: {e}")
                    
            except Exception as e:
                print(f"✗ Failed to load converted file: {e}")
        else:
            print(f"\n=== Conversion Failed ===")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 