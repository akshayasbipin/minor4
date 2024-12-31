import pandas as pd
import numpy as np
import re
import string
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

warnings.filterwarnings('ignore')
nltk.download('stopwords')

@st.cache_data
def load_data():
    df = pd.read_csv('zomato.csv')  
    df = df.drop(['url', 'dish_liked', 'phone'], axis=1)
    df.drop_duplicates(inplace=True)
    df.dropna(how='any', inplace=True)
    df.rename(columns={
        'approx_cost(for two people)': 'cost',
        'listed_in(type)': 'type',
        'listed_in(city)': 'city'
    }, inplace=True)
    df['cost'] = df['cost'].astype(str).str.replace(',', '.').astype(float)

    
    df = df[df['cost'] < 1000]
    
    df = df.loc[df['rate'] != 'NEW']
    df = df.loc[df['rate'] != '-']
    df['rate'] = df['rate'].str.replace('/5', '').str.strip().astype(float)
    df['name'] = df['name'].str.title()
    df['online_order'] = df['online_order'].replace({'Yes': True, 'No': False})
    df['book_table'] = df['book_table'].replace({'Yes': True, 'No': False})

    
    df['Mean Rating'] = df.groupby('name')['rate'].transform('mean')
    scaler = MinMaxScaler(feature_range=(1, 5))
    df[['Mean Rating']] = scaler.fit_transform(df[['Mean Rating']]).round(2)

    
    df = df.drop(['address', 'rest_type', 'type', 'menu_item', 'votes'], axis=1)
    
    
    df = df.sample(n=5000, random_state=42) 
    
    return df


def preprocess_text(df):
    df['reviews_list'] = df['reviews_list'].fillna('').str.lower()
    df['reviews_list'] = df['reviews_list'].apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation))
    )
    stopwords = set(nltk.corpus.stopwords.words('english'))
    df['reviews_list'] = df['reviews_list'].apply(
        lambda x: " ".join([word for word in str(x).split() if word not in stopwords])
    )
    df['reviews_list'] = df['reviews_list'].apply(
        lambda x: re.sub(r'https?://\S+|www\.\S+', '', x)
    )
    return df


def recommend(name, cosine_similarities, indices, df):
    try:
        idx = indices[indices.str.lower() == name.lower()].index[0] 
        score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
        top30_indexes = list(score_series.iloc[1:31].index)  

        recommend_restaurant = []
        for each in top30_indexes:
            recommend_restaurant.append(df.index[each])

        df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])

        for each in recommend_restaurant:
            info = df[['cuisines', 'Mean Rating', 'cost']].loc[df.index == each]
            df_new = pd.concat([df_new, info], ignore_index=True)

        df_new = df_new.drop_duplicates().sort_values(by='Mean Rating', ascending=False).head(10)

        df_new.reset_index(drop=True, inplace=True)
        df_new.insert(0, 'Serial Number', range(1, len(df_new) + 1))  

        return df_new
    except IndexError:
        return None

# Streamlit App Layout
st.title("Restaurant Recommendation App")
df = load_data()
df = preprocess_text(df)

# TF-IDF Matrix
df.set_index('name', inplace=True)
indices = pd.Series(df.index)
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['reviews_list'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

restaurant_name = st.text_input("Enter Restaurant Name to get Recommendations:", "")

if restaurant_name:
    with st.spinner('Searching for recommendations...'):
        recommendations = recommend(restaurant_name, cosine_similarities, indices, df)

    if recommendations is not None and not recommendations.empty:
        st.success(f"Top restaurants similar to '{restaurant_name}':")
        recommendations.reset_index(drop=True, inplace=True)
        st.dataframe(recommendations, width=1200, height=400)
    else:
        st.warning(f"No recommendations found for '{restaurant_name}'. Please try another name.")
