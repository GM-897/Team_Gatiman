from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

from transformers import pipeline
sentiment_pipeline = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')

driver = webdriver.Chrome(options = chrome_options)


titles = []
links = []
driver.get('https://www.hindustantimes.com/')

# Use WebDriverWait to wait for the elements to be present
wait = WebDriverWait(driver, 20)

# Wait for the products to load
wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'htImpressionTracking')))

content = driver.page_source
soup = BeautifulSoup(content, features="html.parser")
for element in soup.findAll('div', attrs={'class': 'cartHolder bigCart track timeAgo'}):
    model = element.find('h3', attrs={'class': 'hdg3'})
    title = model.find('a').text
    link = model.find('a')['href']
    link = 'https://www.hindustantimes.com' + link
    titles.append(title)
    links.append(link)
for element in soup.findAll('div', attrs={'class': 'cartHolder listView track timeAgo'}):
    model = element.find('h3', attrs={'class': 'hdg3'})
    title = model.find('a').text
    link = model.find('a')['href']
    link = 'https://www.hindustantimes.com' + link
    titles.append(title)
    links.append(link)

driver.quit()

# #sentiment analysis on titles
sentiments = sentiment_pipeline(titles)
# #output lists to csv
df = pd.DataFrame({'Title': titles, 'Sentiment': sentiments, 'Link': links})
df.to_csv('results.csv', index=False, encoding='utf-8')
