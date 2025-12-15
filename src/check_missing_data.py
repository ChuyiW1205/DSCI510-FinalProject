#!/usr/bin/env python3
"""
WebScrape Insights project missing data analysis module.
This module is responsible for checking and analyzing missing data in the dataset.
"""
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load processed data
    
    Args:
        filepath (str): Data file path
        
    Returns:
        pd.DataFrame: Data frame
    """
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} does not exist")
        raise FileNotFoundError(f"File {filepath} does not exist")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data, total {len(df)} rows {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def check_missing_data(df: pd.DataFrame) -> dict:
    """
    Check and analyze missing data in the dataset
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Missing data analysis results
    """
    # Calculate missing data statistics
    missing_stats = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Missing Percentage', ascending=False)
    
    # Add data type information
    missing_stats['Data Type'] = df.dtypes.astype(str)
    
    # Filter to only show columns with missing data
    missing_stats = missing_stats[missing_stats['Missing Count'] > 0]
    
    # Overall statistics
    total_cells = np.prod(df.shape)
    total_missing = df.isnull().sum().sum()
    overall_missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
    # Convert to serializable types
    missing_data_details = []
    for _, row in missing_stats.reset_index().rename(columns={'index': 'Column'}).iterrows():
        missing_data_details.append({
            'Column': row['Column'],
            'Missing Count': int(row['Missing Count']),
            'Missing Percentage': float(row['Missing Percentage']),
            'Data Type': row['Data Type']
        })
    
    analysis_results = {
        'missing_columns_count': int(len(missing_stats)),
        'total_columns': int(len(df.columns)),
        'missing_data_details': missing_data_details,
        'overall_missing_stats': {
            'total_cells': int(total_cells),
            'total_missing_cells': int(total_missing),
            'overall_missing_percentage': float(overall_missing_percentage)
        }
    }
    
    logger.info("Missing data analysis completed")
    return analysis_results


def plot_missing_data_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot a heatmap showing the missing data pattern
    
    Args:
        df (pd.DataFrame): Data frame
        output_dir (str): Output directory
    """
    if df.empty:
        logger.info("No data available for missing data heatmap")
        return
    
    # Create a boolean dataframe showing where data is missing
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Data Pattern Heatmap')
    plt.xlabel('Columns')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'missing_data_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Missing data heatmap saved to {output_path}")


def save_missing_data_report(analysis_results: dict, filepath: str) -> None:
    """
    Save missing data analysis report
    
    Args:
        analysis_results (dict): Analysis results
        filepath (str): Save path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save as JSON format
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Missing data analysis report saved to {filepath}")


def print_missing_data_summary(analysis_results: dict) -> None:
    """
    Print a summary of missing data analysis
    
    Args:
        analysis_results (dict): Analysis results
    """
    print("\n=== Missing Data Analysis Summary ===")
    print(f"Total columns in dataset: {analysis_results['total_columns']}")
    print(f"Columns with missing data: {analysis_results['missing_columns_count']}")
    print(f"Overall missing data: {analysis_results['overall_missing_stats']['overall_missing_percentage']:.2f}%")
    
    if analysis_results['missing_data_details']:
        print("\nColumns with missing data:")
        print("-" * 60)
        for item in analysis_results['missing_data_details']:
            print(f"{item['Column']:<30} {item['Missing Count']:<10} ({item['Missing Percentage']:.2f}%)")
    else:
        print("\nNo missing data found in the dataset!")


def main(input_filepath: str = '../data/processed/movies_cleaned.csv',
         output_filepath: str = '../results/missing_data_report.json',
         output_dir: str = '../results/') -> None:
    """
    Main function: Execute missing data analysis process
    
    Args:
        input_filepath (str): Input file path
        output_filepath (str): Output file path for report
        output_dir (str): Output directory for plots
    """
    logger.info("Starting missing data analysis process")
    
    # 1. Load processed data
    df = load_processed_data(input_filepath)
    
    # 2. Check missing data
    analysis_results = check_missing_data(df)
    
    # 3. Save analysis report
    save_missing_data_report(analysis_results, output_filepath)
    
    # 4. Create visualization
    os.makedirs(output_dir, exist_ok=True)
    plot_missing_data_heatmap(df, output_dir)
    
    # 5. Print summary
    print_missing_data_summary(analysis_results)
    
    logger.info("Missing data analysis process completed")


if __name__ == "__main__":
    main()