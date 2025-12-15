#!/usr/bin/env python3
"""
WebScrape Insights project data visualization module.
This module is responsible for generating charts to display analysis results.
"""
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import numpy as np

# Configure Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_analysis_results(filepath: str) -> dict:
    """
    Load analysis results
    
    Args:
        filepath (str): Analysis results file path
        
    Returns:
        dict: Analysis results
    """
    if not os.path.exists(filepath):
        logger.error(f"文件 {filepath} 不存在")
        raise FileNotFoundError(f"文件 {filepath} 不存在")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info("Successfully loaded analysis results")
        return results
    except Exception as e:
        logger.error(f"Error loading analysis results: {e}")
        raise


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load processed data
    
    Args:
        filepath (str): Data file path
        
    Returns:
        pd.DataFrame: Data frame
    """
    if not os.path.exists(filepath):
        logger.error(f"文件 {filepath} 不存在")
        raise FileNotFoundError(f"文件 {filepath} 不存在")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data, total {len(df)} rows {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def plot_score_distribution(score_data: dict, output_dir: str) -> None:
    """
    Plot score distribution chart
    
    Args:
        score_data (dict): Score data
        output_dir (str): Output directory
    """
    if not score_data:
        logger.info("No score data available for visualization")
        return
    
    fig, axes = plt.subplots(len(score_data), 1, figsize=(10, 5*len(score_data)))
    if len(score_data) == 1:
        axes = [axes]
    
    for idx, (score_type, stats) in enumerate(score_data.items()):
        # Create score distribution histogram
        ax = axes[idx] if len(score_data) > 1 else axes[0]
        ax.bar(range(len(stats)), list(stats.values()), color='skyblue')
        ax.set_xlabel('Score Metrics')
        ax.set_ylabel('Values')
        ax.set_title(f'{score_type} Score Statistics')
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(list(stats.keys()), rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(stats.values()):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'score_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Score distribution chart saved to {output_path}")


def plot_genre_distribution(genre_data: dict, output_dir: str) -> None:
    """
    Plot genre distribution chart
    
    Args:
        genre_data (dict): Genre data
        output_dir (str): Output directory
    """
    if not genre_data or 'top_genres' not in genre_data:
        logger.info("No genre data available for visualization")
        return
    
    top_genres = genre_data['top_genres']
    genres = list(top_genres.keys())
    counts = list(top_genres.values())
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(genres, counts, color='lightcoral')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genre')
    plt.title('Movie Genre Distribution (Top 10)')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width}', 
                 ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'genre_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Genre distribution chart saved to {output_path}")


def plot_company_distribution(company_data: dict, output_dir: str) -> None:
    """
    Plot production company distribution chart
    
    Args:
        company_data (dict): Production company data
        output_dir (str): Output directory
    """
    if not company_data or 'top_companies' not in company_data:
        logger.info("No production company data available for visualization")
        return
    
    top_companies = company_data['top_companies']
    companies = list(top_companies.keys())
    counts = list(top_companies.values())
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(companies, counts, color='lightgreen')
    plt.xlabel('Number of Movies')
    plt.ylabel('Production Company')
    plt.title('Production Company Distribution (Top 10)')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width}', 
                 ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'company_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Production company distribution chart saved to {output_path}")


def plot_year_distribution(year_data: dict, output_dir: str) -> None:
    """
    Plot year distribution chart
    
    Args:
        year_data (dict): Year data
        output_dir (str): Output directory
    """
    if not year_data or 'year_distribution' not in year_data:
        logger.info("No year data available for visualization")
        return
    
    year_dist = year_data['year_distribution']
    years = list(map(int, year_dist.keys()))
    counts = list(year_dist.values())
    
    # Create line chart
    plt.figure(figsize=(15, 8))
    plt.plot(years, counts, marker='o', linewidth=2, markersize=6, color='darkblue')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.title('Movie Release Year Distribution')
    plt.grid(True, alpha=0.3)
    
    # Annotate some key points
    if years:
        # Annotate maximum point
        max_idx = np.argmax(counts)
        plt.annotate(f'Max: {counts[max_idx]} movies', 
                    xy=(years[max_idx], counts[max_idx]), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'year_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Year distribution chart saved to {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot correlation heatmap
    
    Args:
        df (pd.DataFrame): Data frame
        output_dir (str): Output directory
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        logger.info("Not enough numeric data to plot correlation heatmap")
        return
    
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Movie Data Correlation Heatmap')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Correlation heatmap saved to {output_path}")


def plot_score_consistency(df: pd.DataFrame, output_dir: str) -> None:
    """
    Draw a score consistency analysis chart
    
    Args:
        df (pd.DataFrame): Data frame
        output_dir (str): Output directory
    """
    if 'critic_reviews_overall_score' not in df.columns or 'user_reviews_overall_score' not in df.columns:
        logger.info("The necessary rating columns are missing, making it impossible to draw a consistency graph")
        return
    
    valid_data = df[['critic_reviews_overall_score', 'user_reviews_overall_score']].dropna()
    
    if len(valid_data) == 0:
        return
    
    # Standardized film critics' ratings
    critic_normalized = valid_data['critic_reviews_overall_score'] / 10
    user_score = valid_data['user_reviews_overall_score']
    
    # create chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Figure 1: The scatter plot shows the correlation
    axes[0].scatter(critic_normalized, user_score, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    
    z = np.polyfit(critic_normalized, user_score, 1)
    p = np.poly1d(z)
    axes[0].plot(critic_normalized, p(critic_normalized), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    
    axes[0].plot([0, 10], [0, 10], 'g--', alpha=0.5, linewidth=1, label='Perfect Agreement')
    
    axes[0].set_xlabel('Critic Score (Normalized 0-10)', fontsize=12)
    axes[0].set_ylabel('User Score (0-10)', fontsize=12)
    axes[0].set_title('Critic vs User Score Correlation', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    
    # Figure 2: Histogram of Differential distribution
    score_diff = critic_normalized - user_score
    axes[1].hist(score_diff, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(score_diff.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean Difference: {score_diff.mean():.2f}')
    axes[1].axvline(0, color='green', linestyle='-', linewidth=1, alpha=0.5, label='No Difference')
    
    axes[1].set_xlabel('Score Difference (Critic - User)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Score Differences', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'score_consistency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"The score consistency chart is saved to {output_path}")


def plot_outliers(analysis_results: dict, output_dir: str) -> None:
    """
    outlier analysis
    
    Args:
        analysis_results (dict): analysis results
        output_dir (str): Output directory
    """
    outlier_data = analysis_results.get('outlier_analysis', {})
    
    if not outlier_data or 'top_disagreements' not in outlier_data:
        logger.info("There is no outlier data available for visualization")
        return
    
    top_outliers = outlier_data['top_disagreements'][:10]
    
    if not top_outliers:
        return
    
    # extraction data
    titles = [movie['title'][:30] + '...' if len(movie['title']) > 30 else movie['title'] 
              for movie in top_outliers]
    critic_scores = [movie['critic_score'] for movie in top_outliers]
    user_scores = [movie['user_score'] for movie in top_outliers]
    differences = [movie['difference'] for movie in top_outliers]
    
    # Create chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Figure 1: Comparison bar chart
    x = np.arange(len(titles))
    width = 0.35
    
    bars1 = axes[0].barh(x - width/2, critic_scores, width, label='Critic Score', color='orange', alpha=0.8)
    bars2 = axes[0].barh(x + width/2, user_scores, width, label='User Score', color='steelblue', alpha=0.8)
    
    axes[0].set_ylabel('Movies', fontsize=12)
    axes[0].set_xlabel('Score (0-10)', fontsize=12)
    axes[0].set_title('Top 10 Movies with Biggest Score Disagreement', fontsize=14, fontweight='bold')
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(titles, fontsize=9)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # Figure 2: Bar chart of difference values
    colors = ['red' if d > 0 else 'blue' for d in differences]
    bars = axes[1].barh(titles, differences, color=colors, alpha=0.7, edgecolor='black')
    
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set_xlabel('Score Difference (Critic - User)', fontsize=12)
    axes[1].set_ylabel('Movies', fontsize=12)
    axes[1].set_title('Score Difference Direction', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        axes[1].text(diff, i, f'{diff:+.2f}', va='center', 
                    ha='left' if diff > 0 else 'right', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'outlier_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"The outlier analysis chart is saved to {output_path}")


def plot_genre_patterns(analysis_results: dict, output_dir: str) -> None:
    """
    Draw a type scoring model diagram
    
    Args:
        analysis_results (dict): analysis result
        output_dir (str): Output directory
    """
    genre_data = analysis_results.get('genre_pattern_analysis', {})
    
    if not genre_data or 'genre_disagreement_ranking' not in genre_data:
        logger.info("There is no type pattern data available for visualization")
        return
    
    genre_patterns = genre_data['genre_disagreement_ranking'][:10]
    
    if not genre_patterns:
        return
    
    # extraction data
    genres = [g['genre'] for g in genre_patterns]
    critic_avgs = [g['critic_avg'] for g in genre_patterns]
    user_avgs = [g['user_avg'] for g in genre_patterns]
    differences = [g['avg_difference'] for g in genre_patterns]
    movie_counts = [g['movie_count'] for g in genre_patterns]
    
    # create chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Figure 1: Grouped bar chart
    x = np.arange(len(genres))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, critic_avgs, width, label='Avg Critic Score', 
                        color='orange', alpha=0.8, edgecolor='black')
    bars2 = axes[0].bar(x + width/2, user_avgs, width, label='Avg User Score', 
                        color='steelblue', alpha=0.8, edgecolor='black')
    
    axes[0].set_xlabel('Genre', fontsize=12)
    axes[0].set_ylabel('Average Score (0-10)', fontsize=12)
    axes[0].set_title('Average Scores by Genre (Ranked by Disagreement)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(genres, rotation=45, ha='right', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add a label
    for i, count in enumerate(movie_counts):
        axes[0].text(i, max(critic_avgs[i], user_avgs[i]) + 0.1, f'n={count}', 
                    ha='center', fontsize=8, style='italic')
    
    # Figure 2: Bar chart of average difference
    colors = ['red' if d > 0 else 'blue' for d in differences]
    bars = axes[1].barh(genres, differences, color=colors, alpha=0.7, edgecolor='black')
    
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set_xlabel('Average Score Difference (Critic - User)', fontsize=12)
    axes[1].set_ylabel('Genre', fontsize=12)
    axes[1].set_title('Genre-wise Score Disagreement', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        axes[1].text(diff, i, f'{diff:+.2f}', va='center', 
                    ha='left' if diff > 0 else 'right', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'genre_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"The type schema diagram is saved to {output_path}")


def plot_studio_patterns(analysis_results: dict, output_dir: str) -> None:
    """
    Draw a comparison chart of the studio model
    
    Args:
        analysis_results (dict): analysis result
        output_dir (str): Output directory
    """
    studio_data = analysis_results.get('studio_pattern_analysis', {})
    
    if not studio_data or 'major_studios' not in studio_data or 'independent' not in studio_data:
        logger.info("There is not enough studio data available for visualization")
        return
    
    # extraction data
    categories = ['Major Studios', 'Independent']
    critic_scores = [
        studio_data['major_studios']['avg_critic_score'],
        studio_data['independent']['avg_critic_score']
    ]
    user_scores = [
        studio_data['major_studios']['avg_user_score'],
        studio_data['independent']['avg_user_score']
    ]
    differences = [
        studio_data['major_studios']['avg_difference'],
        studio_data['independent']['avg_difference']
    ]
    counts = [
        studio_data['major_studios']['count'],
        studio_data['independent']['count']
    ]
    
    # create chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Figure 1: Grouped bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, critic_scores, width, label='Avg Critic Score', 
                        color='orange', alpha=0.8, edgecolor='black')
    bars2 = axes[0].bar(x + width/2, user_scores, width, label='Avg User Score', 
                        color='steelblue', alpha=0.8, edgecolor='black')
    
    axes[0].set_ylabel('Average Score (0-10)', fontsize=12)
    axes[0].set_title('Major Studios vs Independent Films: Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 10)
    
    # Add numerical labels
    for i, (c_score, u_score, count) in enumerate(zip(critic_scores, user_scores, counts)):
        axes[0].text(i - width/2, c_score + 0.1, f'{c_score:.2f}', ha='center', fontsize=9, fontweight='bold')
        axes[0].text(i + width/2, u_score + 0.1, f'{u_score:.2f}', ha='center', fontsize=9, fontweight='bold')
        axes[0].text(i, -0.5, f'n={count}', ha='center', fontsize=9, style='italic')
    
    # Figure 2: Difference Comparison
    colors = ['red' if d > 0 else 'blue' for d in differences]
    bars = axes[1].bar(categories, differences, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_ylabel('Average Score Difference (Critic - User)', fontsize=12)
    axes[1].set_title('Score Disagreement: Major Studios vs Independent', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add numerical labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        axes[1].text(i, diff + 0.05 if diff > 0 else diff - 0.05, f'{diff:+.2f}', 
                    ha='center', va='bottom' if diff > 0 else 'top', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'studio_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"The studio model diagram is saved to {output_path}")


def plot_temporal_trends(analysis_results: dict, output_dir: str) -> None:
    """
    Draw a time trend chart
    
    Args:
        analysis_results (dict): analysis result
        output_dir (str): Output directory
    """
    temporal_data = analysis_results.get('temporal_trend_analysis', {})
    
    if not temporal_data or 'era_analysis' not in temporal_data:
        logger.info("There is no time trend data available for visualization")
        return
    
    era_data = temporal_data['era_analysis']
    
    if not era_data:
        return
    
    # extraction data
    eras = [e['era'] for e in era_data]
    critic_avgs = [e['avg_critic_score'] for e in era_data]
    user_avgs = [e['avg_user_score'] for e in era_data]
    differences = [e['avg_difference'] for e in era_data]
    movie_counts = [e['movie_count'] for e in era_data]
    
    # chart
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Figure 1: Scores change over time
    x = np.arange(len(eras))
    
    axes[0].plot(x, critic_avgs, marker='o', linewidth=2, markersize=8, 
                label='Critic Score', color='orange')
    axes[0].plot(x, user_avgs, marker='s', linewidth=2, markersize=8, 
                label='User Score', color='steelblue')
    
    axes[0].set_xlabel('Era', fontsize=12)
    axes[0].set_ylabel('Average Score (0-10)', fontsize=12)
    axes[0].set_title('Score Trends Across Different Eras', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(eras, fontsize=10)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 10)
    
    
    for i, count in enumerate(movie_counts):
        axes[0].text(i, max(critic_avgs[i], user_avgs[i]) + 0.2, f'n={count}', 
                    ha='center', fontsize=9, style='italic')
    
    # Figure 2: Differences change over time
    colors = ['red' if d > 0 else 'blue' for d in differences]
    bars = axes[1].bar(eras, differences, color=colors, alpha=0.7, edgecolor='black')
    
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_xlabel('Era', fontsize=12)
    axes[1].set_ylabel('Average Score Difference (Critic - User)', fontsize=12)
    axes[1].set_title('Score Disagreement Trends Over Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add numerical labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        axes[1].text(i, diff + 0.05 if diff > 0 else diff - 0.05, f'{diff:+.2f}', 
                    ha='center', va='bottom' if diff > 0 else 'top', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'temporal_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Save the time trend chart to {output_path}")


def create_visualizations(analysis_results: dict, df: pd.DataFrame, output_dir: str = '../results/') -> None:
    """
    Create all visualization charts
    
    Args:
        analysis_results (dict): Analysis results
        df (pd.DataFrame): Data frame
        output_dir (str): Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot score distribution chart
    plot_score_distribution(analysis_results.get('score_analysis', {}), output_dir)
    
    # 2. Plot genre distribution chart
    plot_genre_distribution(analysis_results.get('genre_analysis', {}), output_dir)
    
    # 3. Plot production company distribution chart
    plot_company_distribution(analysis_results.get('company_analysis', {}), output_dir)
    
    # 4. Plot year distribution chart
    plot_year_distribution(analysis_results.get('year_analysis', {}), output_dir)
    
    # 5. Plot correlation heatmap
    plot_correlation_heatmap(df, output_dir)
    
    # 6. in-depth analysis visualization
    plot_score_consistency(df, output_dir)
    plot_outliers(analysis_results, output_dir)
    plot_genre_patterns(analysis_results, output_dir)
    plot_studio_patterns(analysis_results, output_dir)
    plot_temporal_trends(analysis_results, output_dir)
    
    logger.info("All visualization charts have been generated")


def main(analysis_filepath: str = '../results/analysis_results.json',
         data_filepath: str = '../data/processed/movies_cleaned.csv',
         output_dir: str = '../results/') -> None:
    """
    Main function: Execute data visualization process
    
    Args:
        analysis_filepath (str): Analysis results file path
        data_filepath (str): Data file path
        output_dir (str): Output directory
    """
    logger.info("Starting data visualization process")
    
    # 1. Load analysis results
    analysis_results = load_analysis_results(analysis_filepath)
    
    # 2. Load processed data
    df = load_processed_data(data_filepath)
    
    # 3. Create visualization charts
    create_visualizations(analysis_results, df, output_dir)
    
    logger.info("Data visualization process completed")


if __name__ == "__main__":
    main()