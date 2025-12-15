#!/usr/bin/env python3
"""
WebScrape Insights project data collection module.
This module scrapes book information and content from www.metacritic.com
"""
import random
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, urlparse
import logging
import re

from utils.Webpage_parsing import get_director_and_writer, get_actors, get_critic_reviews, get_movie_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# Base URL of the website
BASE_URL = "https://www.metacritic.com/browse/movie/?releaseYearMin=1910&releaseYearMax={}&page={}"

# Create a global Session object for connection reuse
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})


def save_raw_html(html_content, filename):
    """
    Save raw HTML content to data/raw directory
    
    Args:
        html_content (str): HTML content to save
        filename (str): Filename for the HTML file
    """
    raw_dir = '../data/raw'
    os.makedirs(raw_dir, exist_ok=True)
    
    filepath = os.path.join(raw_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"Saved raw HTML to {filepath}")


def get_movie_basic_info():
    """
    Get all main categories from the website
    """
    # Page number
    PAGE_NUM = 1
    # Get current year
    current_year = time.strftime("%Y")

    page_nums = [1,1,10]
    # Get total pages
    while PAGE_NUM <= page_nums[-1]:
        url = BASE_URL.format(current_year, PAGE_NUM)
        try:
            response = session.get(url, allow_redirects=True, timeout=30)
            response.raise_for_status()
            # Check if there is redirect history
            if response.history:
                logger.info(f"Main page request redirected, number of redirects: {len(response.history)}")
                for i, resp in enumerate(response.history):
                    logger.info(f"Redirect {i + 1}: {resp.status_code} -> {resp.url}")
                logger.info(f"Final URL: {response.url}")

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Save raw HTML of main page
            save_raw_html(response.text, f'main_page_{current_year}_page_{PAGE_NUM}.html')

            # class = wid-box bg-white

            # class = c-navigationPagination u-flexbox u-flexbox-alignCenter u-flexbox-justifyCenter g-outer-spacing-top-large g-outer-spacing-bottom-large u-text-center c-pageBrowse__smaller-bottom-spacing
            pagination = soup.find('div', class_='c-navigationPagination')
            # "c-navigationPagination_itemButtonContent u-flexbox u-flexbox-alignCenter u-flexbox-justifyCenter"
            if pagination:
                page_nums_list = [i.text for i in pagination.find_all('span', class_='c-navigationPagination_itemButtonContent u-flexbox u-flexbox-alignCenter u-flexbox-justifyCenter')]
                page_nums_text = "".join(page_nums_list)
                # 正则提取数字
                page_nums_result = re.findall(r'\d+', page_nums_text)  # 修改变量名避免与模块名冲突
                if page_nums_result:
                    page_nums = [int(num) for num in page_nums_result]
            ProductCards = soup.find_all('div', class_='c-finderProductCard')

            for ProductCard in ProductCards:
                # Get html content
                html = ProductCard.prettify()
                # Create a dictionary that returns all information based on the webpage content of cs.html
                product_info = {}

                # Extract movie link
                link_elem = ProductCard.find('a', class_='c-finderProductCard_container')
                if link_elem and link_elem.get('href'):
                    link=link_elem.get('href')
                    # Process into correct link
                    product_info['link'] = urljoin(BASE_URL, link)

                # Extract movie title
                title_elem = ProductCard.find('div', class_='c-finderProductCard_title')
                if title_elem:
                    title_text = title_elem.get_text(strip=True)
                    product_info['title'] = title_text

                # Extract movie rating
                score_elem = ProductCard.find('div', class_='c-siteReviewScore')
                if score_elem:
                    score_text = score_elem.get_text(strip=True)
                    # Keep only the numeric part as the rating
                    score_match = re.search(r'\d+', score_text)  # Using regex module here
                    if score_match:
                        product_info['score'] = int(score_match.group())

                # Extract movie release information (date and rating)
                meta_elem = ProductCard.find('div', class_='c-finderProductCard_meta')
                if meta_elem:
                    meta_spans = meta_elem.find_all('span')
                    if len(meta_spans) >= 3:
                        # First span is release date
                        product_info['release_date'] = meta_spans[0].get_text(strip=True)
                        # Third span is rating information
                        product_info['rating'] = meta_spans[2].get_text(strip=True)

                # Extract movie description information
                desc_elem = ProductCard.find('div', class_='c-finderProductCard_description')
                if desc_elem:
                    product_info['description'] = desc_elem.get_text(strip=True)

                # Print extracted information for verification
                yield product_info
            PAGE_NUM+=1
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get main category page {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error occurred while parsing main category page: {e}")
            raise

#



def get_movie_details_info(url):
    """
    Get movie details information
    """


    response = session.get(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Save raw HTML of movie details page
    url_path = urlparse(url).path
    filename = f"movie_details_{url_path.replace('/', '_')}.html"
    save_raw_html(response.text, filename)

    # Get director and writer
    directors, writers = get_director_and_writer(soup)
    # Get actor list
    actors_list = get_actors(soup)


    # Get review data
    critic_reviews,user_reviews = get_critic_reviews(soup)

    # Get all production company information

    # c-movieDetails_sectionContainer g-inner-spacing-medium u-flexbox u-flexbox-row

    # Get movie:[production companies, release date, duration, rating, genres, website], awards: [xx1,xx2,xx3]
    movie_info = get_movie_info(soup)
    
    # Integrate all detailed information
    movie_detail_info = {}
    movie_detail_info['directors'] = directors
    movie_detail_info['writers'] = writers
    movie_detail_info['actors'] = actors_list
    movie_detail_info['critic_reviews'] = critic_reviews
    movie_detail_info['user_reviews'] = user_reviews
    movie_detail_info.update(movie_info)
    
    return movie_detail_info


def save_processed_movie(movie_data, filename='../data/processed/movies.json'):
    """
    Save processed movie data to file
    
    Args:
        movie_data (dict): Movie data dictionary
        filename (str): Save file path
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # If file exists, load existing data
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                movies = json.load(f)
            except json.JSONDecodeError:
                movies = []
    else:
        movies = []
    
    # Add new movie data
    movies.append(movie_data)
    
    # Save back to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
    return movies


def is_movie_processed(title,filename='../data/processed/movies.json'):
    """
    Check if movie has already been processed (based on director information)
    
    Args:
        directors (str): Movie director information
        filename (str): File path to save processed movie data
        
    Returns:
        bool: Return True if movie has been processed, otherwise return False
    """
    if not os.path.exists(filename):
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            movies = json.load(f)
        
        # Check if director already exists in processed movies
        for movie in movies:
            if movie.get('directors',None) and movie.get('title',None)==title:
                return True
        return False
    except (json.JSONDecodeError, KeyError, Exception):
        return False


def main(max_movies=None):
    """
    Main function: Scrape movie information and save
    
    Args:
        max_movies (int, optional): Maximum number of movies to scrape, None means scrape all movies
    """

    filename='../data/processed/movies.json'
    # Check if exists
    if not os.path.exists(filename):
        movies = []
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            movies = json.load(f)
    if movies:
        logger.info(f"Loaded {len(movies)} processed movies")

    if len(movies)>= max_movies:
        logger.info(f"Number of processed movies reached maximum limit {max_movies}, stop scraping")
        return

    
    for movie_info in get_movie_basic_info():
        # Check if maximum number of movies is reached
        if max_movies is not None and len(movies) >= max_movies:
            logger.info(f"Reached set maximum number of movies {max_movies}, stop scraping")
            break
        
        link = movie_info['link']
        # Check if movie has already been processed (based on director information)
        if is_movie_processed(movie_info['title']):
            logger.info(f"Movie {movie_info.get('title', 'Unknown')} already processed, skipping")
            continue
        # Get movie details
        logger.info(f"Processing movie: {movie_info.get('title', 'Unknown')}")
        try:
            movie_details = get_movie_details_info(link)
            movie_info.update(movie_details)
        except Exception as e:
            logger.error(f"Error getting details for movie {movie_info.get('title', 'Unknown')}: {e}")


        time.sleep(random.uniform(1, 3.5))
        # Save movie information
        movies = save_processed_movie(movie_info)
        logger.info(f"Processed and saved movie {len(movies) + 1}: {movie_info.get('title', 'Unknown')}")


if __name__ == '__main__':
    # Set maximum number of movies to scrape, for example 10 movies

    main(max_movies=200)
