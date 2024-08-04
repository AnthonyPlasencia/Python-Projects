#pandas: Used for creating and manipulating data frames to store the scraped data
import pandas as pd

#numpy: Used for numerical operations and data manipulation
import numpy as np

#requests: Used for sending HTTP requests to the website and getting the response
import requests as req

#BeautifulSoup: Used for parsing the HTML content of the website
import bs4 as bs

#matplotlib: Used for displaying the word cloud
import matplotlib.pyplot as plt

#wordcloud: Used for generating the word cloud visualization.
from wordcloud import WordCloud

# Define the URL to scrape
url = "http://quotes.toscrape.com"

# Send a GET request to the URL and get the response
response = req.get(url)

# Create a BeautifulSoup object to parse the HTML content
soup = bs.BeautifulSoup(response.text, "html.parser")

# Initialize empty lists to store the scraped data
authors = []
quotes = []
tags = []

# Find all quote elements on the page
items = soup.find_all("div", class_="quote")

# Loop through each quote element and extract the necessary information
for element in items:
    quotes.append(element.find("span", class_="text").text)
    authors.append(element.find("small", class_="author").text)

    # Find all tags for each quote
    tag = element.findAll("a", {"class": "tag"})
    tagList = []
    for i in tag:
        tagList.append(i.text)
    tags.append(tagList)

# Join the tags into a single string separated by commas
tags = [','.join(tags[i]) for i in range(len(tags))]

# Create a pandas DataFrame to store the scraped data
df = pd.DataFrame({
    'Author': authors,
    'Quotes': quotes,
    'Tags': tags
})

# Save the DataFrame to a CSV file
df.to_csv('WebScrape.csv', index=False)

# Generate a word cloud from the tags
words = " ".join(tags)
wordcloud = WordCloud(background_color="white", max_font_size=200, min_font_size=10, height=1200, width=1600, colormap="rainbow").generate(words)

# Display the word cloud
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Word Cloud of Tags", fontsize=25, fontweight='bold', pad=20)
plt.axis("off")
plt.show()
