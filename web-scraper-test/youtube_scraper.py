import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline

# Disable parallelism in Transformers to avoid issues with TensorFlow
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Chrome WebDriver options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')

# Define the sentiment analysis pipeline
sentiment_pipeline = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Define the list of YouTube channel URLs to monitor
channel_urls = [
    'https://www.youtube.com/@aajtak/videos',
    'https://www.youtube.com/@zeenews/videos',
    # Add more channel URLs as needed
]

# Check if the CSV file exists, and create it if not
csv_filename = 'yt_results.csv'
if not os.path.exists(csv_filename):
    df = pd.DataFrame(columns=['Title', 'Sentiment', 'Link'])
    df.to_csv(csv_filename, index=False, encoding='utf-8')

# Initialize the last scraped titles
last_scraped_titles = []

# Set the scraping interval (in seconds)
scraping_interval = 600  # 10 minutes

print("Starting YouTube scraper...")

while True:
    for channel_url in channel_urls:
        print(f"Scraping {channel_url}...")
        # Initialize lists to store scraped data
        titles = []
        links = []


        # Navigate to the YouTube channel URL

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(channel_url)

        # Wait for the page to load
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_all_elements_located((By.ID, 'thumbnail')))

        # Parse the page source
        content = driver.page_source
        soup = BeautifulSoup(content, features="html.parser")

        # Extract video titles and links
        for element in soup.findAll('a', attrs={'id': 'video-title-link'}):
            title = element.text
            link = element['href']
            link = 'https://www.youtube.com' + link
            titles.append(title)
            links.append(link)

        # Close the WebDriver
        driver.quit()

        # Filter out new titles that were not in the last scrape
        new_titles = list(set(titles) - set(last_scraped_titles))

        # Skip if no new titles were found
        if(new_titles == []):
            print("No new videos found")
            continue

        #find subtitles for all video links

        # Perform sentiment analysis on new titles
        sentiments = sentiment_pipeline(new_titles)
        
        # Update the last scraped titles
        last_scraped_titles.extend(new_titles)

        # Print and update the CSV file
        if new_titles:
            print(f'{new_titles.__len__()} videos found on {channel_url}:')
            df = pd.DataFrame({'Title': new_titles, 'Sentiment': sentiments, 'Link': links})
            df.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8')
        
        driver.quit()

    # Wait for the specified interval before the next scrape.
    print(f"Waiting {scraping_interval/60} minutes before next scrape...")
    time.sleep(scraping_interval)
    print("Starting next scrape...")
    