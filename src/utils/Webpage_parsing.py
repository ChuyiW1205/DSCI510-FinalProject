# 获取导演和编剧
import re
from typing import  Tuple


def get_director_and_writer(soup):
    """
    获取导演和编剧
    """
    director_elem = soup.find('div', class_='c-productDetails_staff')
    director_match = re.search(r'Directed By:\s*(.+?)(?:\n\s*Written By:|$)', director_elem.text, re.DOTALL)
    writer_match = re.search(r'Written By:\s*(.+?)(?:\n\s*$|$)', director_elem.text, re.DOTALL)

    # Clean extracted text, remove excess spaces and line breaks
    if director_match:
        directors = ' '.join(director_match.group(1).split())
    else:
        directors = None

    if writer_match:
        writers = ' '.join(writer_match.group(1).split())
    else:
        writers = None
    return directors, writers
# Get actor list
# Get actor list
def get_actors(soup)-> list[dict[str, str]]:
    # Get actor list c-globalCarousel_content c-globalCarousel_content-scrollable c-globalCarousel_content-scrollable_mobile-gap-small
    actors_elem = soup.find('div',
                            class_='c-globalCarousel_content c-globalCarousel_content-scrollable c-globalCarousel_content-scrollable_mobile-gap-small')
    actors_list = []

    if actors_elem:
        actors = actors_elem.findAll('div', class_='c-globalPersonCard')

        for actor in actors:
            # Extract actor name
            name_element = actor.find('h3', class_='c-globalPersonCard_name')
            actor_name = name_element.text.strip() if name_element else None
            # Extract character name
            role_element = actor.find('h4', class_='c-globalPersonCard_role')
            character_name = role_element.text.strip() if role_element else None

            # Add actor information to list
            actors_list.append({
                'name': actor_name,
                'role': character_name
            })

    return actors_list
def get_critic_reviews(soup)-> tuple[dict, dict]:
    reviews_elems = soup.findAll('div', class_='c-reviewsSummaryHeader c-reviewsSection_header--desktop')

    # Initialize review data dictionary
    critic_reviews = {}
    user_reviews = {}

    if reviews_elems:
        reviews_elem = reviews_elems[0]
        # Extract total review count
        total_reviews_elem = reviews_elem.find('span', string=re.compile(r'Based on \d+ Critic Reviews'))
        if total_reviews_elem:
            total_text = total_reviews_elem.get_text()
            total_match = re.search(r'(\d+)', total_text)
            if total_match:
                critic_reviews['total_count'] = int(total_match.group(1))

        # Extract overall score
        score_elem = reviews_elem.find('div', class_='c-siteReviewScore_large')
        if score_elem:
            score_text = score_elem.get_text(strip=True)
            score_match = re.search(r'\d+', score_text)
            if score_match:
                critic_reviews['overall_score'] = int(score_match.group())

        # Extract positive, neutral, and negative review counts
        stats_elems = reviews_elem.find_all('div', class_=re.compile(r'c-reviewsStats_\w+Stats'))
        for stat_elem in stats_elems:
            stat_text = stat_elem.get_text()
            count_match = re.search(r'(\d+)\s+Reviews', stat_text)
            percent_match = re.search(r'(\d+)%', stat_text)

            if 'positive' in stat_elem.get('class', []) or 'Positive' in stat_text:
                if count_match:
                    critic_reviews['positive_count'] = int(count_match.group(1))
                if percent_match:
                    critic_reviews['positive_percent'] = int(percent_match.group(1))
            elif 'neutral' in stat_elem.get('class', []) or 'Mixed' in stat_text:
                if count_match:
                    critic_reviews['mixed_count'] = int(count_match.group(1))
                if percent_match:
                    critic_reviews['mixed_percent'] = int(percent_match.group(1))
            elif 'negative' in stat_elem.get('class', []) or 'Negative' in stat_text:
                if count_match:
                    critic_reviews['negative_count'] = int(count_match.group(1))
                if percent_match:
                    critic_reviews['negative_percent'] = int(percent_match.group(1))

    # Find user review section

    if reviews_elems:
        user_reviews_section = reviews_elems[1]
        # Find user score summary
        user_score_elem = user_reviews_section.find('div', class_='c-siteReviewScore_user')
        if user_score_elem:
            user_score_text = user_score_elem.get_text(strip=True)
            user_score_match = re.search(r'\d+', user_score_text)
            if user_score_match:
                user_reviews['overall_score'] = float(user_score_match.group())

        # Find total user review count
        user_total_elem = user_reviews_section.find('span', string=re.compile(r'Based on \d+ User Ratings'))
        if user_total_elem:
            user_total_text = user_total_elem.get_text()
            user_total_match = re.search(r'(\d+)', user_total_text)
            if user_total_match:
                user_reviews['total_count'] = int(user_total_match.group(1))

        # Find user positive, neutral, and negative review counts and percentages
        user_stats_elems = user_reviews_section.find_all('div', class_=re.compile(r'c-reviewsStats_\w+Stats'))
        for stat_elem in user_stats_elems:
            stat_text = stat_elem.get_text()
            count_match = re.search(r'(\d+)\s+Ratings', stat_text)
            percent_match = re.search(r'(\d+)%', stat_text)

            if 'positive' in stat_elem.get('class', []) or 'Positive' in stat_text:
                if count_match:
                    user_reviews['positive_count'] = int(count_match.group(1))
                if percent_match:
                    user_reviews['positive_percent'] = int(percent_match.group(1))
            elif 'neutral' in stat_elem.get('class', []) or 'Mixed' in stat_text:
                if count_match:
                    user_reviews['mixed_count'] = int(count_match.group(1))
                if percent_match:
                    user_reviews['mixed_percent'] = int(percent_match.group(1))
            elif 'negative' in stat_elem.get('class', []) or 'Negative' in stat_text:
                if count_match:
                    user_reviews['negative_count'] = int(count_match.group(1))
                if percent_match:
                    user_reviews['negative_percent'] = int(percent_match.group(1))
    return critic_reviews,user_reviews
def get_production_companies(soup):
    production_companies = soup.find('div', class_='c-movieDetails_sectionContainer g-inner-spacing-medium u-flexbox u-flexbox-row')
    # Get company list g-outer-spacing-left-medium-fluid
    production_company_list = production_companies.find_all('li', class_='c-gameDetails_listItem u-inline-block g-color-gray70')

    production_company_list = [i.text.strip() for i in production_company_list]
    return production_company_list

def get_movie_info(soup):
    """
    Extract movie details from soup object

    Parameters:
        soup: BeautifulSoup object containing HTML of movie details page

    Returns:
        dict: Dictionary containing movie information including production companies, release date, duration, rating, genres, website, and awards
    """

    # Initialize movie information dictionary
    movie_info = {
        'production_companies': [],
        'release_date': '',
        'duration': '',
        'rating': '',
        'genres': [],
        'website': '',
        'awards': []
    }

    # Extract movie details container
    movie_details_div = soup.find('div', class_='c-movieDetails')
    if not movie_details_div:
        return movie_info

    # Extract production company information
    prod_company_section = get_production_companies(soup)


    movie_info['production_companies'] = prod_company_section
    # Extract release date
    release_date_section = movie_details_div.find('span', string='Release Date')
    if release_date_section:
        release_date_value = release_date_section.find_next_sibling()
        if release_date_value:
            movie_info['release_date'] = release_date_value.get_text().strip()

    # Extract duration
    duration_section = movie_details_div.find('span', string='Duration')
    if duration_section:
        duration_value = duration_section.find_next_sibling()
        if duration_value:
            movie_info['duration'] = duration_value.get_text().strip()

    # Extract rating
    rating_section = movie_details_div.find('span', string='Rating')
    if rating_section:
        rating_value = rating_section.find_next_sibling()
        if rating_value:
            movie_info['rating'] = rating_value.get_text().strip()

    # Extract genres
    genres_section = movie_details_div.find('span', string='Genres')
    if genres_section:
        genre_list = genres_section.find_next('ul', class_='c-genreList')
        if genre_list:
            genre_items = genre_list.find_all('li')
            for item in genre_items:
                genre_link = item.find('a')
                if genre_link:
                    genre_text = genre_link.get_text().strip()
                    movie_info['genres'].append(genre_text)

    # Extract website
    website_section = movie_details_div.find('span', string='Website')
    if website_section:
        website_value = website_section.find_next('span')
        if website_value:
            website_link = website_value.find('a')
            if website_link and website_link.get('href'):
                movie_info['website'] = website_link.get('href').strip()

    # Extract awards information
    awards_section = soup.find('div', attrs={'data-testid': 'details-award-summary'})
    if awards_section:
        award_items = awards_section.find_all('div', class_='c-productionAwardSummary_award')
        for award in award_items:
            award_name_elem = award.find('div', class_='g-text-bold')
            award_details_elem = award.find('div')
            if award_name_elem and award_details_elem:
                award_name = award_name_elem.get_text().strip()
                award_details = award_details_elem.get_text().strip()
                movie_info['awards'].append(f"{award_name} {award_details}")

    return movie_info