# WebScrape Insights

A complete data science project that covers the full pipeline from web scraping, data cleaning, data analysis, to data visualization.  
This project focuses on collecting and analyzing movie rating data from Metacritic to explore the consistency and differences between critic scores and user scores.

---

## Project Information

**Project Name:** WebScrape Insights  
**Team Member:** Chuyi Wang  
**USC Email:** chuyi@usc.edu  
**USC ID:** 5458365139  

---

## Project Structure

DSCI510-FinalProject/
├── src/
│   ├── get_data.py
│   ├── clean_data.py
│   ├── run_analysis.py
│   └── visualize_results.py
├── data/
│   ├── raw/
│   │   └── movie_html.zip
│   └── processed/
│       ├── movies.json
│       └── movies_cleaned.csv
├── results/
│   └── *.png
├── README.md
├── requirements.txt
├── project_proposal.pdf
├── reply to the proposal question.md
└──  final_report.pdf



---

## Environment Setup

It is recommended to create a virtual environment before running this project to avoid dependency conflicts.

## Notes on Raw Data

Due to GitHub file upload limitations, the raw HTML files scraped from Metacritic are stored  
as a compressed archive in `data/raw/movie_html.zip`.

The raw data is preserved for reference and reproducibility purposes.  
The analysis and visualization scripts do not require the raw HTML files to be manually extracted,  
as all subsequent processing is performed on the cleaned datasets stored in `data/processed/`.


### Using Conda

conda create -n webscrape_env python=3.9
conda activate webscrape_env


Installing Dependencies

After activating the virtual environment, install all required libraries:
pip install -r requirements.txt

Data Collection

Script: get_data.py

This script scrapes movie data from Metacritic using requests and BeautifulSoup.
Movies are collected based on Metacritic rankings and span a wide range of release years.

Output:

Raw HTML files are saved to data/raw/

Scraping supports resume-from-breakpoint to avoid duplicate downloads

Run:
cd src
python get_data.py

Data Cleaning

Script: clean_data.py

This script cleans and preprocesses the raw scraped data by handling missing values, removing duplicates, cleaning HTML tags, and converting the data into a structured tabular format.

Output:

Cleaned dataset saved to data/processed/movies_cleaned.csv

Run:
cd src
python clean_data.py

Missing Data Check

Script: check_missing_data.py

This script checks the cleaned dataset for missing values and generates a summary report and visualization.

Output:

results/missing_data_report.json

results/missing_data_heatmap.png

Run:cd src
python check_missing_data.py

Data Analysis

Script: run_analysis.py

This script performs statistical and exploratory analysis on the cleaned data, including descriptive statistics, score consistency analysis, genre-based analysis, production studio comparison, time trend analysis, and outlier detection.

Output:

Analysis results saved to results/analysis_results.json

Run:
cd src
python run_analysis.py

Data Visualization

Script: visualize_results.py

This script generates visualizations to support the analysis results, including score distributions, correlation heatmaps, score consistency plots, genre and studio comparisons, outlier analysis, and temporal trends.

Output:

All figures are saved as .png files in the results/ directory

Run:cd src
python visualize_results.py


Reproducibility

The project follows a fully reproducible workflow:

Data collection

Data cleaning

Missing data validation

Data analysis

Data visualization

Running the scripts in the above order will reproduce all results used in the final report.

## Notes

For detailed project background, methodology, and conclusions, please refer to
`project_proposal.pdf` and `results/final_report.pdf`.


