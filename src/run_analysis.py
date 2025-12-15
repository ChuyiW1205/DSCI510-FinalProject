#!/usr/bin/env python3
"""
WebScrape Insights project data analysis module.
This module is responsible for analyzing cleaned movie data and generating descriptive statistical results.
"""
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import json

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


def basic_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Statistics information
    """
    stats = {}
    
    # Basic statistics for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    stats['numeric_stats'] = df[numeric_columns].describe().to_dict() if numeric_columns else {}
    
    # Basic statistics for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    stats['categorical_stats'] = {}
    for col in categorical_columns[:10]:  # Limit display to first 10 categorical columns
        value_counts = df[col].value_counts()
        stats['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': value_counts.head(5).to_dict()  # Top 5 most common values
        }
    
    stats['total_rows'] = len(df)
    stats['total_columns'] = len(df.columns)
    
    logger.info("Basic statistics calculation completed")
    return stats


def analyze_scores(df: pd.DataFrame) -> dict:
    """
    Analyze score data
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Score analysis results
    """
    score_analysis = {}
    
    # Find possible score columns
    score_columns = [col for col in df.columns if 'score' in col.lower()]
    
    for col in score_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # Remove null values
            scores = df[col].dropna()
            
            if len(scores) > 0:
                score_analysis[col] = {
                    'mean': float(scores.mean()),
                    'median': float(scores.median()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'count': int(len(scores))
                }
    
    logger.info("Score data analysis completed")
    return score_analysis


def analyze_genres(df: pd.DataFrame) -> dict:
    """
    Analyze movie genre distribution
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Genre analysis results
    """
    genre_analysis = {}
    
    if 'genres' in df.columns:
        # Split genre strings and count
        all_genres = []
        for genres_str in df['genres'].dropna():
            if genres_str != 'None':
                genres_list = [g.strip() for g in genres_str.split(',')]
                all_genres.extend(genres_list)
        
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts()
            genre_analysis = {
                'total_unique_genres': len(genre_counts),
                'top_genres': genre_counts.head(10).to_dict(),  # Top 10 most common genres
                'genre_distribution': genre_counts.to_dict()
            }
    
    logger.info("Genre distribution analysis completed")
    return genre_analysis


def analyze_production_companies(df: pd.DataFrame) -> dict:
    """
    Analyze production company distribution
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Production company analysis results
    """
    company_analysis = {}
    
    if 'production_companies' in df.columns:
        # Split production company strings and count
        all_companies = []
        for companies_str in df['production_companies'].dropna():
            if companies_str != 'None':
                companies_list = [c.strip() for c in companies_str.split(',')]
                all_companies.extend(companies_list)
        
        if all_companies:
            company_counts = pd.Series(all_companies).value_counts()
            company_analysis = {
                'total_unique_companies': len(company_counts),
                'top_companies': company_counts.head(10).to_dict(),  # Top 10 most common companies
                'company_distribution': company_counts.to_dict()
            }
    
    logger.info("Production company distribution analysis completed")
    return company_analysis


def analyze_release_years(df: pd.DataFrame) -> dict:
    """
    Analyze release year distribution
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Year analysis results
    """
    year_analysis = {}
    
    if 'release_date' in df.columns:
        # Extract years
        years = []
        for date_str in df['release_date'].dropna():
            if date_str != 'None':
                # Try to extract year from date string
                year_match = pd.Series(date_str).str.extract(r'(\d{4})')
                if not year_match.empty and not pd.isna(year_match.iloc[0, 0]):
                    years.append(int(year_match.iloc[0, 0]))
        
        if years:
            year_series = pd.Series(years)
            year_analysis = {
                'earliest_year': int(year_series.min()),
                'latest_year': int(year_series.max()),
                'year_distribution': year_series.value_counts().sort_index().to_dict()
            }
    
    logger.info("Release year distribution analysis completed")
    return year_analysis


def analyze_score_consistency(df: pd.DataFrame) -> dict:
    """
    Analyze the consistency of the ratings given by film critics and users
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: The results of the consistency analysis of the scores
    """
    consistency_analysis = {}
    
    # Check whether the necessary columns exist
    if 'critic_reviews_overall_score' not in df.columns or 'user_reviews_overall_score' not in df.columns:
        logger.warning("缺少必要的评分列，无法进行一致性分析")
        return consistency_analysis
    
    # Remove missing values
    valid_data = df[['critic_reviews_overall_score', 'user_reviews_overall_score', 'title']].dropna()
    
    if len(valid_data) == 0:
        logger.warning("没有足够的有效数据进行一致性分析")
        return consistency_analysis
    
    # Standardized film critics' ratings to the range of 0-10 (originally 0-100)
    critic_normalized = valid_data['critic_reviews_overall_score'] / 10
    user_score = valid_data['user_reviews_overall_score']
    
    # variance in calculation
    score_diff = critic_normalized - user_score
    
    # 1. correlation analysis
    from scipy.stats import pearsonr, spearmanr
    pearson_corr, pearson_pvalue = pearsonr(critic_normalized, user_score)
    spearman_corr, spearman_pvalue = spearmanr(critic_normalized, user_score)
    
    consistency_analysis['correlation'] = {
        'pearson_coefficient': float(pearson_corr),
        'pearson_pvalue': float(pearson_pvalue),
        'spearman_coefficient': float(spearman_corr),
        'spearman_pvalue': float(spearman_pvalue),
        'interpretation': 'Strong positive correlation' if pearson_corr > 0.7 else 
                         'Moderate positive correlation' if pearson_corr > 0.4 else
                         'Weak correlation'
    }
    
    # 2. indifference statistics
    consistency_analysis['difference_stats'] = {
        'mean_difference': float(score_diff.mean()),
        'median_difference': float(score_diff.median()),
        'std_difference': float(score_diff.std()),
        'min_difference': float(score_diff.min()),
        'max_difference': float(score_diff.max()),
        'interpretation': 'Critics generally score higher' if score_diff.mean() > 0 else 'Users generally score higher'
    }
    
    # 3.Consistency classification
    highly_consistent = (abs(score_diff) <= 1).sum()
    moderately_consistent = ((abs(score_diff) > 1) & (abs(score_diff) <= 2)).sum()
    inconsistent = (abs(score_diff) > 2).sum()
    
    total = len(score_diff)
    consistency_analysis['consistency_categories'] = {
        'highly_consistent_count': int(highly_consistent),
        'highly_consistent_percent': float(highly_consistent / total * 100),
        'moderately_consistent_count': int(moderately_consistent),
        'moderately_consistent_percent': float(moderately_consistent / total * 100),
        'inconsistent_count': int(inconsistent),
        'inconsistent_percent': float(inconsistent / total * 100)
    }
    
    logger.info("The consistency analysis of the scores has been completed")
    return consistency_analysis


def analyze_outliers(df: pd.DataFrame) -> dict:
    """
    Outlier analysis: Identify the films with the greatest differences in ratings between critics and users
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Outlier analysis results
    """
    outlier_analysis = {}
    
    # Check the necessary columns
    if 'critic_reviews_overall_score' not in df.columns or 'user_reviews_overall_score' not in df.columns:
        logger.warning("The necessary rating columns are missing, making outlier analysis impossible")
        return outlier_analysis
    
    # Preparing the Database Environment
    valid_data = df[['title', 'critic_reviews_overall_score', 'user_reviews_overall_score', 
                     'genres', 'production_companies', 'release_date']].dropna(subset=['critic_reviews_overall_score', 'user_reviews_overall_score'])
    
    if len(valid_data) == 0:
        return outlier_analysis
    
    # Calculate standardized differences
    valid_data = valid_data.copy()
    valid_data['critic_normalized'] = valid_data['critic_reviews_overall_score'] / 10
    valid_data['score_difference'] = valid_data['critic_normalized'] - valid_data['user_reviews_overall_score']
    valid_data['abs_difference'] = abs(valid_data['score_difference'])
    
    # Find out the 10 movies with the greatest differences
    top_outliers = valid_data.nlargest(10, 'abs_difference')
    
    outlier_list = []
    for _, row in top_outliers.iterrows():
        outlier_list.append({
            'title': row['title'],
            'critic_score': float(row['critic_normalized']),
            'user_score': float(row['user_reviews_overall_score']),
            'difference': float(row['score_difference']),
            'abs_difference': float(row['abs_difference']),
            'genres': row['genres'],
            'production_companies': row['production_companies'],
            'release_date': row['release_date']
        })
    
    outlier_analysis['top_disagreements'] = outlier_list
    
    # Analyze the common features of the films with the greatest differences
    # type
    outlier_genres = []
    for genres_str in top_outliers['genres'].dropna():
        if genres_str != 'None':
            outlier_genres.extend([g.strip() for g in genres_str.split(',')])
    
    if outlier_genres:
        genre_counts = pd.Series(outlier_genres).value_counts()
        outlier_analysis['common_genres'] = genre_counts.head(5).to_dict()
    
    # year
    outlier_years = []
    for date_str in top_outliers['release_date'].dropna():
        if date_str != 'None':
            year_match = pd.Series(date_str).str.extract(r'(\d{4})')
            if not year_match.empty and not pd.isna(year_match.iloc[0, 0]):
                outlier_years.append(int(year_match.iloc[0, 0]))
    
    if outlier_years:
        outlier_analysis['year_range'] = {
            'earliest': int(min(outlier_years)),
            'latest': int(max(outlier_years)),
            'mean': float(np.mean(outlier_years))
        }
    
    logger.info("The outlier analysis has been completed")
    return outlier_analysis


def analyze_genre_patterns(df: pd.DataFrame) -> dict:
    """
    Analyze the rating models of different types of movies
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Analysis results of the type scoring model
    """
    genre_pattern_analysis = {}
    
    if 'genres' not in df.columns or 'critic_reviews_overall_score' not in df.columns or 'user_reviews_overall_score' not in df.columns:
        logger.warning("The necessary columns are missing, making type pattern analysis impossible")
        return genre_pattern_analysis
    
    # Expand type data
    genre_scores = []
    for _, row in df.iterrows():
        if pd.notna(row['genres']) and row['genres'] != 'None':
            if pd.notna(row['critic_reviews_overall_score']) and pd.notna(row['user_reviews_overall_score']):
                genres_list = [g.strip() for g in row['genres'].split(',')]
                critic_norm = row['critic_reviews_overall_score'] / 10
                user_score = row['user_reviews_overall_score']
                diff = critic_norm - user_score
                
                for genre in genres_list:
                    genre_scores.append({
                        'genre': genre,
                        'critic_score': critic_norm,
                        'user_score': user_score,
                        'difference': diff
                    })
    
    if not genre_scores:
        return genre_pattern_analysis
    
    genre_df = pd.DataFrame(genre_scores)
    
    # Statistics by type
    genre_stats = genre_df.groupby('genre').agg({
        'critic_score': ['mean', 'count'],
        'user_score': 'mean',
        'difference': ['mean', 'std']
    }).round(3)
    
    # Only retain genres with at least three films
    genre_stats = genre_stats[genre_stats[('critic_score', 'count')] >= 3]
    
    # Identify the type with the greatest differences
    genre_stats['abs_diff'] = abs(genre_stats[('difference', 'mean')])
    genre_stats_sorted = genre_stats.sort_values('abs_diff', ascending=False)
    
    genre_pattern_list = []
    for genre in genre_stats_sorted.index[:10]:
        genre_pattern_list.append({
            'genre': genre,
            'critic_avg': float(genre_stats.loc[genre, ('critic_score', 'mean')]),
            'user_avg': float(genre_stats.loc[genre, ('user_score', 'mean')]),
            'avg_difference': float(genre_stats.loc[genre, ('difference', 'mean')]),
            'std_difference': float(genre_stats.loc[genre, ('difference', 'std')]),
            'movie_count': int(genre_stats.loc[genre, ('critic_score', 'count')])
        })
    
    genre_pattern_analysis['genre_disagreement_ranking'] = genre_pattern_list
    
    logger.info("The analysis of the type scoring model has been completed")
    return genre_pattern_analysis


def analyze_studio_patterns(df: pd.DataFrame) -> dict:
    """
    Analysis of Independent Films vs The rating differences of films from major studios
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Analysis results of the studio model
    """
    studio_analysis = {}
    
    if 'production_companies' not in df.columns:
        logger.warning("The list of production companies is missing, making it impossible to conduct studio analysis")
        return studio_analysis
    
    # Define a major film studio
    major_studios = [
        'Warner Bros.', 'Paramount Pictures', 'Universal', 'Columbia Pictures',
        'Twentieth Century Fox', 'Metro-Goldwyn-Mayer', 'MGM', 'Walt Disney',
        'RKO Radio Pictures'
    ]
    
    # MovieSort
    major_studio_movies = []
    independent_movies = []
    
    for _, row in df.iterrows():
        if pd.notna(row['production_companies']) and row['production_companies'] != 'None':
            if pd.notna(row['critic_reviews_overall_score']) and pd.notna(row['user_reviews_overall_score']):
                companies = row['production_companies']
                is_major = any(studio in companies for studio in major_studios)
                
                critic_norm = row['critic_reviews_overall_score'] / 10
                user_score = row['user_reviews_overall_score']
                diff = critic_norm - user_score
                
                movie_data = {
                    'critic_score': critic_norm,
                    'user_score': user_score,
                    'difference': diff
                }
                
                if is_major:
                    major_studio_movies.append(movie_data)
                else:
                    independent_movies.append(movie_data)
    
    # statistic analysis
    if major_studio_movies:
        major_df = pd.DataFrame(major_studio_movies)
        studio_analysis['major_studios'] = {
            'count': len(major_studio_movies),
            'avg_critic_score': float(major_df['critic_score'].mean()),
            'avg_user_score': float(major_df['user_score'].mean()),
            'avg_difference': float(major_df['difference'].mean()),
            'std_difference': float(major_df['difference'].std())
        }
    
    if independent_movies:
        indie_df = pd.DataFrame(independent_movies)
        studio_analysis['independent'] = {
            'count': len(independent_movies),
            'avg_critic_score': float(indie_df['critic_score'].mean()),
            'avg_user_score': float(indie_df['user_score'].mean()),
            'avg_difference': float(indie_df['difference'].mean()),
            'std_difference': float(indie_df['difference'].std())
        }
    
    # comparative analysis
    if major_studio_movies and independent_movies:
        studio_analysis['comparison'] = {
            'critic_score_diff': float(studio_analysis['major_studios']['avg_critic_score'] - 
                                     studio_analysis['independent']['avg_critic_score']),
            'user_score_diff': float(studio_analysis['major_studios']['avg_user_score'] - 
                                   studio_analysis['independent']['avg_user_score']),
            'interpretation': 'Major studios have higher critic scores' if 
                            studio_analysis['major_studios']['avg_critic_score'] > studio_analysis['independent']['avg_critic_score']
                            else 'Independent films have higher critic scores'
        }
    
    logger.info("The studio model analysis has been completed")
    return studio_analysis


def analyze_temporal_trends(df: pd.DataFrame) -> dict:
    """
    Analyze the changing trend of the score over time
    
    Args:
        df (pd.DataFrame): Data frame
        
    Returns:
        dict: Time trend analysis results
    """
    temporal_analysis = {}
    
    if 'release_date' not in df.columns:
        logger.warning("The release date column is missing, making it impossible to conduct time trend analysis")
        return temporal_analysis
    
    # Extract the year and score
    year_scores = []
    for _, row in df.iterrows():
        if pd.notna(row['release_date']) and row['release_date'] != 'None':
            if pd.notna(row['critic_reviews_overall_score']) and pd.notna(row['user_reviews_overall_score']):
                year_match = pd.Series(row['release_date']).str.extract(r'(\d{4})')
                if not year_match.empty and not pd.isna(year_match.iloc[0, 0]):
                    year = int(year_match.iloc[0, 0])
                    critic_norm = row['critic_reviews_overall_score'] / 10
                    user_score = row['user_reviews_overall_score']
                    
                    year_scores.append({
                        'year': year,
                        'critic_score': critic_norm,
                        'user_score': user_score,
                        'difference': critic_norm - user_score
                    })
    
    if not year_scores:
        return temporal_analysis
    
    year_df = pd.DataFrame(year_scores)
    
    # Group by age (every 20 years in one group)
    def get_era(year):
        if year < 1950:
            return 'Pre-1950'
        elif year < 1970:
            return '1950-1969'
        elif year < 1990:
            return '1970-1989'
        elif year < 2010:
            return '1990-2009'
        else:
            return '2010+'
    
    year_df['era'] = year_df['year'].apply(get_era)
    
    era_stats = year_df.groupby('era').agg({
        'critic_score': 'mean',
        'user_score': 'mean',
        'difference': ['mean', 'std'],
        'year': 'count'
    }).round(3)
    
    era_list = []
    for era in ['Pre-1950', '1950-1969', '1970-1989', '1990-2009', '2010+']:
        if era in era_stats.index:
            era_list.append({
                'era': era,
                'avg_critic_score': float(era_stats.loc[era, ('critic_score', 'mean')]),
                'avg_user_score': float(era_stats.loc[era, ('user_score', 'mean')]),
                'avg_difference': float(era_stats.loc[era, ('difference', 'mean')]),
                'std_difference': float(era_stats.loc[era, ('difference', 'std')]),
                'movie_count': int(era_stats.loc[era, ('year', 'count')])
            })
    
    temporal_analysis['era_analysis'] = era_list
    
    logger.info("The time trend analysis has been completed")
    return temporal_analysis


def save_analysis_results(results: dict, filepath: str) -> None:
    """
    Save analysis results
    
    Args:
        results (dict): Analysis results
        filepath (str): Save path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save as JSON format
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Analysis results saved to {filepath}")


def main(input_filepath: str = '../data/processed/movies_cleaned.csv',
         output_filepath: str = '../results/analysis_results.json') -> None:
    """
    Main function: Execute data analysis process
    
    Args:
        input_filepath (str): Input file path
        output_filepath (str): Output file path
    """
    logger.info("Starting data analysis process")
    
    # 1. Load processed data
    df = load_processed_data(input_filepath)
    
    # 2. Calculate basic statistics
    basic_stats = basic_statistics(df)
    
    # 3. Analyze score data
    score_stats = analyze_scores(df)
    
    # 4. Analyze genre distribution
    genre_stats = analyze_genres(df)
    
    # 5. Analyze production company distribution
    company_stats = analyze_production_companies(df)
    
    # 6. Analyze release year distribution
    year_stats = analyze_release_years(df)
    
    # 7. 新增深度分析
    consistency_stats = analyze_score_consistency(df)
    outlier_stats = analyze_outliers(df)
    genre_pattern_stats = analyze_genre_patterns(df)
    studio_pattern_stats = analyze_studio_patterns(df)
    temporal_trend_stats = analyze_temporal_trends(df)
    
    # 8. Integrate all analysis results
    analysis_results = {
        'basic_statistics': basic_stats,
        'score_analysis': score_stats,
        'genre_analysis': genre_stats,
        'company_analysis': company_stats,
        'year_analysis': year_stats,
        'consistency_analysis': consistency_stats,
        'outlier_analysis': outlier_stats,
        'genre_pattern_analysis': genre_pattern_stats,
        'studio_pattern_analysis': studio_pattern_stats,
        'temporal_trend_analysis': temporal_trend_stats
    }
    
    # 9. Save analysis results
    save_analysis_results(analysis_results, output_filepath)
    
    logger.info("Data analysis process completed")
    
    # Print some key results
    print("\n=== Data Analysis Overview ===")
    print(f"Total records: {basic_stats['total_rows']}")
    print(f"Total fields: {basic_stats['total_columns']}")
    
    if score_stats:
        print("\n=== Score Analysis ===")
        for score_type, stats in score_stats.items():
            print(f"{score_type}: Average={stats['mean']:.2f}, Max={stats['max']}, Min={stats['min']}")
    
    if genre_stats:
        print("\n=== Genre Analysis ===")
        print(f"Total {genre_stats['total_unique_genres']} genres")
        print("Most common genres:")
        for genre, count in list(genre_stats['top_genres'].items())[:5]:
            print(f"  {genre}: {count} movies")
    
    if company_stats:
        print("\n=== Production Company Analysis ===")
        print(f"Total {company_stats['total_unique_companies']} production companies")
        print("Most common production companies:")
        for company, count in list(company_stats['top_companies'].items())[:5]:
            print(f"  {company}: {count} movies")
    
    # Print the newly added in-depth analysis results
    if consistency_stats and 'correlation' in consistency_stats:
        print("\n=== Score consistency analysis ===")
        print(f"Pearson correlation coefficient: {consistency_stats['correlation']['pearson_coefficient']:.3f}")
        print(f"Explanation of relevance: {consistency_stats['correlation']['interpretation']}")
        print(f"Average score difference: {consistency_stats['difference_stats']['mean_difference']:.2f}")
        print(f"Consistency classification:")
        print(f"  Highly consistent: {consistency_stats['consistency_categories']['highly_consistent_percent']:.1f}%")
        print(f"  Moderate consistency: {consistency_stats['consistency_categories']['moderately_consistent_percent']:.1f}%")
        print(f"  inconsistency: {consistency_stats['consistency_categories']['inconsistent_percent']:.1f}%")
    
    if outlier_stats and 'top_disagreements' in outlier_stats:
        print("\n=== Outlier Analysis (Movies with the Greatest Differences) ===")
        for i, movie in enumerate(outlier_stats['top_disagreements'][:5], 1):
            print(f"{i}. {movie['title']}")
            print(f"   movie critic: {movie['critic_score']:.1f} | user: {movie['user_score']:.1f} | difference: {movie['difference']:+.2f}")
    
    if genre_pattern_stats and 'genre_disagreement_ranking' in genre_pattern_stats:
        print("\n=== Type scoring model (the type with the greatest disagreement) ===")
        for i, genre in enumerate(genre_pattern_stats['genre_disagreement_ranking'][:5], 1):
            print(f"{i}. {genre['genre']} (n={genre['movie_count']})")
            print(f"   mean difference: {genre['avg_difference']:+.2f} (movie critic: {genre['critic_avg']:.1f}, user: {genre['user_avg']:.1f})")
    
    if studio_pattern_stats and 'comparison' in studio_pattern_stats:
        print("\n=== Analysis of the Studio Model ===")
        print(f"Big studio films: {studio_pattern_stats['major_studios']['count']} ")
        print(f"  The average score of film critics: {studio_pattern_stats['major_studios']['avg_critic_score']:.2f}")
        print(f"  Average score of users: {studio_pattern_stats['major_studios']['avg_user_score']:.2f}")
        print(f"independent film: {studio_pattern_stats['independent']['count']} ")
        print(f"  The average score of film critics: {studio_pattern_stats['independent']['avg_critic_score']:.2f}")
        print(f"  Average score of users: {studio_pattern_stats['independent']['avg_user_score']:.2f}")
    
    if temporal_trend_stats and 'era_analysis' in temporal_trend_stats:
        print("\n=== time-trend analysis ===")
        for era_data in temporal_trend_stats['era_analysis']:
            print(f"{era_data['era']}: Average difference {era_data['avg_difference']:+.2f} (n={era_data['movie_count']})")


if __name__ == "__main__":
    main()