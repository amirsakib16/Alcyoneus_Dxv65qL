"""
Run this script locally to optimize your pickle file size
This creates a smaller version with only essential data
"""
import pickle
import pandas as pd
import os

def optimize_pickle(input_file, output_file):
    """
    Optimize pickle file by:
    1. Keeping only necessary columns
    2. Using efficient data types
    3. Removing duplicates
    """
    print(f"Loading original pickle file: {input_file}")
    
    # Check original file size
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"Original file size: {original_size:.2f} MB")
    
    # Load the dataframe
    with open(input_file, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Original dataframe shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Keep only necessary columns
    essential_columns = ['title', 'TKN']
    
    # Check which columns exist
    available_columns = [col for col in essential_columns if col in df.columns]
    
    if not available_columns:
        print("ERROR: Required columns not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    df_optimized = df[available_columns].copy()
    
    # Remove duplicates based on title
    if 'title' in df_optimized.columns:
        df_optimized = df_optimized.drop_duplicates(subset=['title'], keep='first')
    
    # Convert string columns to category if they have repeated values
    for col in df_optimized.columns:
        if df_optimized[col].dtype == 'object':
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:  # If less than 50% unique
                df_optimized[col] = df_optimized[col].astype('category')
    
    print(f"Optimized dataframe shape: {df_optimized.shape}")
    
    # Save optimized pickle
    with open(output_file, 'wb') as f:
        pickle.dump(df_optimized, f, protocol=4)
    
    # Check new file size
    new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"New file size: {new_size:.2f} MB")
    print(f"Size reduction: {((original_size - new_size) / original_size * 100):.1f}%")
    
    if new_size > 200:
        print("\n⚠️  WARNING: File is still too large for Vercel!")
        print("Consider:")
        print("1. Using external storage (GitHub Releases, Vercel Blob)")
        print("2. Further reducing the dataset")
        print("3. Using a database instead of pickle")

if __name__ == "__main__":
    input_file = "information.pkl"
    output_file = "information_optimized.pkl"
    
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found!")
        print("Please make sure the file is in the same directory as this script.")
    else:
        optimize_pickle(input_file, output_file)
        print(f"\n✅ Optimized file saved as: {output_file}")
        print("Replace your original file with this one before deploying.")