import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

# Ensure NLTK Stopwords are downloaded
nltk.download('stopwords')

# Title of the Streamlit app
st.title("Restaurant Recommendation App")

# Try reading the dataset and handle any errors gracefully
try:
    df1 = pd.read_csv('zomato.csv')
    st.write("Data loaded successfully!")
except Exception as e:
    st.write("Error loading data:", e)

# Data cleaning and preprocessing
zomato = df1.drop(['url', 'dish_liked', 'phone'], axis=1)
zomato.drop_duplicates(inplace=True)
zomato.dropna(how='any', inplace=True)
zomato = zomato.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})
zomato['cost'] = zomato['cost'].apply(lambda x: str(x).replace(',', '.')).astype(float)

# Step 1: Remove entries with 'NEW' and '-'
zomato = zomato.loc[zomato['rate'] != 'NEW']
zomato = zomato.loc[zomato['rate'] != '-'].reset_index(drop=True)

# Step 2: Define a lambda function to clean up the rates
remove_slash = lambda x: x.replace('/5', '') if isinstance(x, str) else x

# Step 3: Apply the function, strip whitespace, and convert to float
zomato['rate'] = zomato['rate'].apply(remove_slash).str.strip().astype(float)

# Adjust the column names
zomato['name'] = zomato['name'].apply(lambda x: x.title())
zomato['online_order'].replace(('Yes','No'),(True, False),inplace=True)
zomato['book_table'].replace(('Yes','No'),(True, False),inplace=True)

# Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0
for restaurant in restaurants:
    zomato.loc[zomato['name'] == restaurant, 'Mean Rating'] = zomato[zomato['name'] == restaurant]['rate'].mean()

# Scale the Mean Rating to a range from 1 to 5
scaler = MinMaxScaler(feature_range=(1, 5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

# Text processing for reviews
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].str.lower()
zomato["reviews_list"] = zomato["reviews_list"].apply(remove_punctuation)
zomato["reviews_list"] = zomato["reviews_list"].apply(remove_stopwords)
zomato["reviews_list"] = zomato["reviews_list"].apply(remove_urls)

# Generating Recommendations with TF-IDF and Cosine Similarity
df_percent = zomato.sample(frac=0.5)
df_percent.set_index('name', inplace=True)

# Compute the TF-IDF matrix for reviews
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_percent.index)

def recommend(name, cosine_similarities=cosine_similarities):
    recommend_restaurant = []

    # Find the index of the restaurant entered
    idx = indices[indices == name].index[0]

    # Find the restaurants with a similar cosine similarity value and order them from highest to lowest
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)

    # Extract top 30 restaurant indexes with a similar cosine similarity value
    top30_indexes = list(score_series.iloc[0:31].index)

    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])

    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])

    # Collect information for top 30 restaurants and avoid duplicates
    for each in recommend_restaurant:
        restaurant_info = df_percent[['cuisines', 'Mean Rating', 'cost']][df_percent.index == each]
        df_new = pd.concat([df_new, restaurant_info.sample(n=min(1, len(restaurant_info)))], ignore_index=True)

    # Drop duplicates and sort the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep='first')
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    return df_new

# UI for user input
restaurant_name = st.text_input('Enter Restaurant Name to get Recommendations: ', '')

if restaurant_name:
    st.write(f"Searching for restaurants similar to '{restaurant_name}'...")

    with st.spinner('Processing your request, please wait...'):
        try:
            recommended = recommend(restaurant_name)
            if not recommended.empty:
                st.write("Here are some similar restaurants based on reviews:")
                st.dataframe(recommended)
            else:
                st.write("No recommendations found.")
        except Exception as e:
            st.write("Error in recommendation:", e)
else:
    st.write("Please enter a restaurant name to get recommendations.")