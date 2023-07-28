#!/usr/bin/env python
# coding: utf-8

# In[31]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load the dataset
df = pd.read_csv('Drugdata.csv')
df.dropna(axis=0,inplace=True)
df=df.dropna()

# Define the conditions we want to focus on
CONDITIONS = {
    'Depression': ['depression'],
    'High Blood Pressure': ['hypertension', 'high blood pressure'],
    'Diabetes, Type 2': ['diabetes, type 2']
}

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

from bs4 import BeautifulSoup
import re
import string
def review_to_words(raw_review):
    # delete html 
    review_text = BeautifulSoup(raw_review,'html.parser').get_text()
    # make a space
    latters_only = re.sub('[^a-zA-Z]', ' ',review_text)
    # lower letters
    words = latters_only.lower().split()
    # stop words
    meaningful_words = [w for w in words if not w in stop]
    # lemmitization
    lemmatize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    #space join words
    return(' '.join(lemmatize_words))

# Define a function to get the top recommended drugs for a given condition
def get_top_drugs(condition, num=5):
    # Filter the dataset for the specified condition
    condition_keywords = CONDITIONS[condition]
    condition_reviews = df[df['condition'].str.contains('|'.join(condition_keywords), case=False)]
    
    # Train a TF-IDF vectorizer on the drug reviews
    vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=100, max_df= 1.0, stop_words='english')
    drug_vectors = vectorizer.fit_transform(condition_reviews['review'])
    
    # Calculate cosine similarities between the drug vectors
    similarities = cosine_similarity(drug_vectors)
    
    # Get the indices of the top-rated drugs
    ratings = condition_reviews['rating']
    top_indices = ratings.nlargest(num).index
    
    # Get the drug names and corresponding condition reviews
    top_drugs = condition_reviews.iloc[top_indices]['drugName']
    top_reviews = condition_reviews.iloc[top_indices]['review']
    
    return top_drugs, top_reviews

# Define the Streamlit app
st.title('Drug Recommender')

condition = st.selectbox('Select a condition to find recommended drugs:', list(CONDITIONS.keys()))

if st.button('Find drugs'):
    top_drugs, top_reviews = get_top_drugs(condition)
    st.header('Top recommended drugs:')
    for drug, review in zip(top_drugs, top_reviews):
        st.subheader(drug)
        st.write(review)

