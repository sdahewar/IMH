"""
Data loading utilities for the Insights Engine
"""

import os
import pandas as pd
from typing import Optional


def load_data(filepath: str = "data/Data Voice Hackathon_Master.xlsx") -> Optional[pd.DataFrame]:
    """
    Load the raw Excel data
    
    Args:
        filepath: Path to the Excel file
        
    Returns:
        DataFrame with the loaded data or None if failed
    """
    # Try multiple paths
    paths_to_try = [
        filepath,
        "Data Voice Hackathon_Master.xlsx",
        "data/Data Voice Hackathon_Master.xlsx",
        "../Data Voice Hackathon_Master.xlsx"
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path)
                print(f"Loaded {len(df):,} records from {path}")
                return df
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    print(f"Could not find data file. Tried: {paths_to_try}")
    return None


def load_classified_data(filepath: str = "output/classified_calls.csv") -> Optional[pd.DataFrame]:
    """
    Load classified data from CSV
    
    Args:
        filepath: Path to the classified CSV file
        
    Returns:
        DataFrame with classified data or None if not found
    """
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df):,} classified records from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading classified data: {e}")
    
    return None


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has required columns
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_cols = ['transcript', 'customer_type', 'city_name', 'call_duration']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"Warning: Missing columns: {missing}")
        return False
    
    return True

