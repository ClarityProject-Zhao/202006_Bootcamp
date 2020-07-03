from sys import path
from bs4 import BeautifulSoup, Comment
import requests
import pandas as pd
import numpy as np
import os
import re
import string
import matplotlib.pyplot as plt

def Scrape(base_url, url):
    df_output = pd.DataFrame()
    cols = ["title", "location", "length", "rating", "rating_number", "price"]

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    block_select = soup.find_all("div", class_="_1xl0u0x")
    for tag in block_select:
        # processing each experience

        title = tag.find("div", {"class": "_1qusvl7"}).text
        # Extract title of the experience
        if not title.startswith('COMING SOON'):
            # Extract all the other attributes
            location = tag.find("div", {"class": "_1ycij1l"}).text
            price_hour_set = tag.find_all("li", {"class": "_g86r3e"})
            price = price_hour_set[0].text
            hour_raw = price_hour_set[1].text
            hour = hour_raw[3:len(hour_raw)]

            rating_ratingNum_set = tag.find("span", {"class": "_krjbj"})
            if not (rating_ratingNum_set is None):
                rating = rating_ratingNum_set.contents[0]
                ratingNum_raw = rating_ratingNum_set.contents
                ratingNum = ratingNum_raw[len(ratingNum_raw) - 1]
            else:
                rating = np.nan
                ratingNum = np.nan

            # Create a new entry, insert into the dictionary, drop duplicate
            new_data = (title, location, hour, rating, ratingNum, price)
            new_df = pd.DataFrame([new_data], columns=cols)
            if df_output.empty:
                df_output = new_df
            else:
                df_output = df_output.append(new_df)

    # After collecting all experience, find the link to the next page
    # For pages that are neither first page nor last, the search should return two lines for 'previous' and 'next'
    # First page will return only one line 'next' and last page will only return one line 'first'
    Page_raw_set = soup.find_all("a", {"class", "_4si2sk3"})
    Page_raw = Page_raw_set[len(Page_raw_set) - 1]

    if Page_raw['aria-label'] == 'Next page':
        nextPage_url = base_url + Page_raw['href']
        nextPage_flag = True
    else:
        # elseif: Page_raw['aria-label']=='Previous page':
        nextPage_url = ''
        nextPage_flag = False

    return df_output, nextPage_flag, nextPage_url

# Scraping the first page
base_url = "https://www.airbnb.com"
initial_url = 'https://www.airbnb.com/s/experiences/online'
airbnb_virtualExp, nextPage_flag, nextPage_url = Scrape(base_url, initial_url)

pg = 0
# data_backup={}
# data_backup[0]=airbnb_virtualExp
# next_url_backup={}
# next_url_backup[0] = nextPage_url

# Continuing scraping the following pages
while nextPage_flag:
    pg = pg + 1
    print(f'processing pg {pg}..')
    new_experience, nextPage_flag, nextPage_url = Scrape(base_url, nextPage_url)
    airbnb_virtualExp = airbnb_virtualExp.append(new_experience)
    # data_backup[pg] = new_experience
    # next_url_backup[pg] = nextPage_url

##output data
path = os.path.abspath(os.getcwd())
file_name = os.path.join(path, 'data.csv')
airbnb_virtualExp.to_csv(file_name, index=False)


##cleaning data
df = airbnb_virtualExp.copy()
df['length'] = airbnb_virtualExp['length'].apply(lambda x: x.replace(" hours", "").replace(" hour", "") if isinstance(x, str) else x).astype(float)
df['rating'] = airbnb_virtualExp['rating'].apply(lambda x: x.replace("Rating ", "").replace(" out of 5", "") if isinstance(x, str) else x).astype(float)
df['rating_number'] = airbnb_virtualExp['rating_number'].apply(lambda x: x.replace(" reviews", "").replace(" review", "") if isinstance(x, str) else x).astype(float)
df['price'] = airbnb_virtualExp['price'].apply(lambda x: x.replace("From $", "").replace("/person", "") if isinstance(x, str) else x).astype(float)

df = df.drop_duplicates()


##plotting data
df_byLoc=pd.DataFrame()
agg_byLoc=df[['rating','rating_number','length','price','location']].groupby('location')
df_byLoc['Avg_Price']=agg_byLoc['price'].agg(np.nanmean)
df_byLoc['Count_Experience']=agg_byLoc['price'].count()
df_byLoc['Median_Rating']=agg_byLoc['rating'].agg(np.nanmedian)
df_byLoc['Avg_Length']=agg_byLoc['length'].agg(np.nanmean)
df_byLoc['Avg_Rating']=agg_byLoc['rating'].agg(np.nanmean)
df_byLoc['Median_Rating_Num']=agg_byLoc['rating_number'].agg(np.median)
df_byLoc['Avg_Rating_Num']=agg_byLoc['rating_number'].agg(np.mean)

df_out=df_byLoc.sort_values(by='Count_Experience',ascending=False)
df_out.head(10).plot.bar(rot=0, y=['Avg_Price'],figsize=(20,5))
plt.show()
