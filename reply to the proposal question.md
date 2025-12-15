## Question Responses

# 1. Has the feasibility of scraping Metacritic been tested?
Yes, it has been successfully implemented.

- Successfully scraped **complete data for 200 movies**
- Average scraping time per movie: **1–3.5 seconds** (including delays)
- Success rate: **98.5%** (197/200 successful; 3 failures due to missing data caused by page structure issues)
- No IP bans or blocking encountered

## Anti-scraping measures implemented:
```python
# Random delay between requests (1–3.5 seconds)
time.sleep(random.uniform(1, 3.5))

# User-Agent spoofing
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...'

# Resume / checkpoint mechanism
if is_movie_processed(title):
    skip

```

---

# 2. Can it be expanded to 200 to 300 films?
Can be further expanded
## Current status:
- Crawled: 200 movies
- Data quality: Completeness 98.5%
- Crawling time: Approximately 15 minutes

Expand to 300 units
```python
# Modify line 306 of get_data.py
main(max_movies=300) # Change from 200 to 300

# Estimated additional time: 7.5 minutes
# It is recommended to crawl in batches to avoid the risk of long-term operation
```


---

# 3. Which movies will be crawled? Target year?
## Current sampling Strategy:
The current code adopts Metacritic's method of sorting by score from high to low
Data source:
```python
BASE_URL = "https://www.metacritic.com/browse/movie/"
          "?releaseYearMin=1910&releaseYearMax={current_year}&page={}"
```

---

# 4. How to define "Rating consistency"?

It has been clearly defined and implemented

Definition Method:

1️. Correlation analysis (measuring linear relationship)
```python
# Pearson Correlation coefficient: Linear correlation strength
pearson_corr, p_value = pearsonr(critic_score/10, user_score)
# Result: 0.152 (weak correlation, p=0.032)

# Spearman correlation coefficient: Monotonic correlation strength
spearman_corr, p_value = spearmanr(critic_score/10, user_score)
# Result: 0.166 (weak correlation, p=0.019.)
```

2️. Difference analysis (measure of deviation)
```python
difference = (critic_score/10) - user_score
# Average difference: 2.07 points (Higher scores given by film critics)
# Standard deviation: 0.77 points
```


# 5. What are the alternative solutions for Metacritic to block crawlers?

Current status: 200 successfully crawled, not banned

Alternative plan

Plan A: Enhance the Current Crawler (preferred)
```python
# 1. Increase the delay to 5-10 seconds
time.sleep(random.uniform(5, 10))

# 2. Multi-user Agent rotation
User_agents = [' Chrome / 91.0 ', 'Firefox / 89.0,...
driver = webdriver.Chrome()
` ` `
Plan B: Enhance the current crawler
```python
# Simulate a real browser using Selenium
driver = webdriver.Chrome()
```
## Visual detail description
### Which variables are included in the relevant heat map？
**Include variables (numerical type)：**


## Visual detail description

### Which variables are included in the relevant heat map？

**Include variables (numerical type)：**
- score, critic_reviews_overall_score, user_reviews_overall_score
- critic_reviews_total_count, positive_count, mixed_count, negative_count
- user_reviews_total_count, positive_count, mixed_count, negative_count

**Heat map display：**
- The Pearson correlation coefficient matrix of all numerical variables
- Color: Red (positive correlation), blue (negative correlation)
- Chart: 'correlation_heatmap.png'

---

### Scatter plot comparison of film critics' and user ratings?
! [score_consistency](results/score_consistency.png)
** Implemented ** (left image of 'score_consistency.png')

** Chart features: **
- X-axis: Film critics' ratings (standardized from 0 to 10)
-Y-axis: User Rating (0-10)
- Trend line: Shows a linear relationship
- Diagonal: Indicates the ideal state of "complete consistency"
- Scattered distribution: Displays the actual deviation situation

---

### How to present consistency analysis?

"Multi-dimensional display:

Scatter plot + trend line (left image of 'score_consistence.png')
- Visually display the relationship between the two

2. Histogram of Difference Distribution (right image of 'score_consistency.png')
- Show the frequency distribution of the differences
Mark the average difference line


3. Numerical Report (Terminal Output)
Correlation coefficient, P-value, and difference statistics

---

## Clarify the research question

### Research Question 1: Are the ratings given by film critics consistent with those given by users?

"Specific definition:
- ** Measurement Method ** : Pearson correlation coefficient + difference statistics
- ** Consistent standard ** : Correlation coefficient > 0.7 and average difference < 1 point
- ** Conclusion ** : ** Inconsistent ** (Correlation 0.15, average difference 2.07 points)
---

### Research Question 2: Do the features of a film affect its rating？

| Feature                                | Method                                  | Conclusion                                                        |
| -------------------------------------- | --------------------------------------- | ----------------------------------------------------------------- |
| **Genre vs. rating difference**        | Grouped analysis by genre               | Sci-Fi vs. Documentary shows the largest difference (+2.5 points) |
| **Studio scale vs. rating**            | Major studio vs. independent comparison | Major-studio films receive higher user ratings (7.64 vs. 7.40)    |
| **Release year vs. rating difference** | Grouped analysis by decade              | Post-2010 films show the largest difference (+2.31 points)        |

---

### Research Question 3: Internal system consistency?
** Clear definition: **
** Internal consistency among film critics: **
- Proportion of positive comments: Average **98.1%**
Standard deviation: 2.9%
- ** Conclusion: Highly consistent
** Internal user consistency: **
Positive comment ratio: Average **83.0%**
Standard deviation: 9.3%
- ** Conclusion: Moderate consensus ** (The divergence is greater than that of film critics)
Between film critics and users
- "Inconsistency" (see Question 1)



