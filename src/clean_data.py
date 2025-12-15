#!/usr/bin/env python3
"""
WebScrape Insights project data cleaning module.
This module is responsible for cleaning and preprocessing raw data scraped from websites.
"""
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import pandas as pd
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import data scraping module
from get_data import get_movie_details_info, save_processed_movie


def load_raw_data(filepath: str) -> List[Dict[Any, Any]]:
    """
    Load raw data
    
    Args:
        filepath (str): Raw data file path
        
    Returns:
        List[Dict]: Raw data list
    """
    if not os.path.exists(filepath):
        logger.warning(f"File {filepath} does not exist")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} raw data entries")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        return []


def clean_text(text: str) -> str:
    """
    Clean text data, remove excess whitespace characters and HTML tags
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Remove excess whitespace characters
    clean = re.sub(r'\s+', ' ', clean)
    # Remove leading and trailing spaces
    clean = clean.strip()
    
    return clean


def identify_missing_fields(item: Dict) -> List[str]:
    """
    Identify missing fields in data item
    
    Args:
        item (Dict): Data item
        
    Returns:
        List[str]: Missing fields list
    """
    missing_fields = []
    
    # Define key fields
    key_fields = ['directors', 'writers', 'actors', 'critic_reviews', 'user_reviews', 
                  'production_companies', 'duration', 'genres', 'awards']
    
    for field in key_fields:
        value = item.get(field)
        # Check if field is missing
        if value is None or (isinstance(value, str) and (not value.strip() or value == "Unknown")):
            missing_fields.append(field)
        elif isinstance(value, list) and len(value) == 0:
            missing_fields.append(field)
        elif isinstance(value, dict) and len(value) == 0:
            missing_fields.append(field)
            
    return missing_fields


def update_missing_data(data: List[Dict]) -> List[Dict]:
    """
    Update missing data, try to re-scrape
    
    Args:
        data (List[Dict]): Raw data list
        
    Returns:
        List[Dict]: Updated data list
    """
    updated_data = []
    updated_count = 0
    
    for item in data:
        # Identify missing fields
        missing_fields = identify_missing_fields(item)
        
        # If there are missing fields and a link, try to re-scrape
        if missing_fields and 'link' in item and item['link']:
            logger.info(f"Movie '{item.get('title', 'Unknown')}' has missing fields: {missing_fields}, trying to re-scrape")
            try:
                # Get movie details
                movie_details = get_movie_details_info(item['link'])
                
                # Update missing fields
                for field in missing_fields:
                    if field in movie_details:
                        item[field] = movie_details[field]
                        logger.info(f"Successfully updated field '{field}'")
                
                updated_count += 1
            except Exception as e:
                logger.warning(f"Error re-scraping movie '{item.get('title', 'Unknown')}': {e}")
        
        updated_data.append(item)
    
    logger.info(f"Successfully updated {updated_count} data")
    return updated_data


def handle_missing_values(data: List[Dict]) -> List[Dict]:
    """
    Handle remaining missing values (data that cannot be obtained through re-scraping)
    
    Args:
        data (List[Dict]): Data list
        
    Returns:
        List[Dict]: Data list after handling missing values
    """
    cleaned_data = []
    
    for item in data:
        # Create new dictionary to avoid modifying original data
        cleaned_item = {}
        
        for key, value in item.items():
            # Handle various types of missing values
            if value is None:
                cleaned_item[key] = "Unknown"
            elif isinstance(value, str) and (not value.strip() or value == "Unknown"):
                cleaned_item[key] = "Unknown"
            elif isinstance(value, list) and len(value) == 0:
                cleaned_item[key] = []
            elif isinstance(value, dict) and len(value) == 0:
                cleaned_item[key] = {}
            else:
                cleaned_item[key] = value
                
        cleaned_data.append(cleaned_item)
    
    logger.info("Missing value processing completed")
    return cleaned_data


def remove_duplicates(data: List[Dict]) -> List[Dict]:
    """
    Remove duplicates (based on movie title)
    
    Args:
        data (List[Dict]): Data list
        
    Returns:
        List[Dict]: Deduplicated data list
    """
    seen_titles = set()
    unique_data = []
    
    for item in data:
        title = item.get('title', '')
        if title not in seen_titles:
            seen_titles.add(title)
            unique_data.append(item)
    
    logger.info(f"Before deduplication: {len(data)} data entries, After deduplication: {len(unique_data)} data entries")
    return unique_data


def convert_to_structured_format(data: List[Dict]) -> pd.DataFrame:
    """
    Convert data to structured format (DataFrame)
    
    Args:
        data (List[Dict]): Cleaned data list
        
    Returns:
        pd.DataFrame: Structured data
    """
    
    flattened_data = []
    
    for item in data:
        flattened_item = {}
        
        # Copy top-level fields
        for key, value in item.items():
            if not isinstance(value, (dict, list)):
                flattened_item[key] = value if value is not None else "Unknown"
            elif isinstance(value, list) and key in ['production_companies', 'genres', 'awards']:
                # Convert list to string
                flattened_item[key] = ', '.join(str(v) for v in value) if value else "None"
            elif isinstance(value, dict) and key in ['critic_reviews', 'user_reviews']:
                # Flatten review data
                for sub_key, sub_value in value.items():
                    flattened_item[f"{key}_{sub_key}"] = sub_value if sub_value is not None else "Unknown"
            else:
                flattened_item[key] = str(value) if value else "None"
                
        flattened_data.append(flattened_item)
    
    df = pd.DataFrame(flattened_data)
    logger.info(f"The data has been converted to a structured format {len(df)} row {len(df.columns)} column")
    return df


def handle_dataframe_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in DataFrame
    
    Args:
        df (pd.DataFrame): Original DataFrame
        
    Returns:
        pd.DataFrame: DataFrame after handling missing values
    """
    # For string columns, fill missing values with "Unknown"
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].fillna("Unknown")
    
    # For numeric columns, fill missing values with median
    numeric_columns = df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    logger.info("DataFrame missing value processing completed")
    return df


def translate_to_english(text: str) -> str:
    """
    Translate non-English text to English (placeholder function)
    
    Args:
        text (str): Original text
        
    Returns:
        str: Translated English text
    """
    return text


def save_processed_data(data: pd.DataFrame, filepath: str) -> None:
    """
    Save processed data
    
    Args:
        data (pd.DataFrame): Processed data
        filepath (str): Save path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save as CSV format
    data.to_csv(filepath, index=False, encoding='utf-8')
    logger.info(f"Data saved to {filepath}")


def main(input_filepath: str = '../data/processed/movies.json', 
         output_filepath: str = '../data/processed/movies_cleaned.csv') -> None:
    """
    Main function: Execute data cleaning process
    
    Args:
        input_filepath (str): Input file path
        output_filepath (str): Output file path
    """
    logger.info("Starting data cleaning process")
    
    # 1. Load raw data
    raw_data = load_raw_data(input_filepath)
    if not raw_data:
        logger.warning("No raw data found, ending cleaning process")
        return
    
    # 2. Try to update missing data (re-scraping)
    updated_data = update_missing_data(raw_data)
    
    # 3. Handle remaining missing values
    data_without_missing = handle_missing_values(updated_data)
    
    # 4. Remove duplicates
    unique_data = remove_duplicates(data_without_missing)
    
    # 5. Clean text data
    for item in unique_data:
        for key, value in item.items():
            if isinstance(value, str):
                item[key] = clean_text(value)
    
    # 6. Convert to structured format
    structured_data = convert_to_structured_format(unique_data)
    
    # 7. Handle missing values in DataFrame
    structured_data = handle_dataframe_missing_values(structured_data)
    
    # 8. Save processed data
    save_processed_data(structured_data, output_filepath)
    
    logger.info("Data cleaning process completed")


if __name__ == "__main__":
    main()