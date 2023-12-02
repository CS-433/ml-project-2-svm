# I tried to group all the commands that should be ran no matter what 

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from utilities import *
import pandas as pd
import numpy as np 
import json

# pick full or smaller version of dataset
df = pd.read_csv('data/modelready_220423.csv')
# df = pd.read_csv('data/ten_percent.csv')

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
print(f'tot columns = {len(df.columns)}, numeric type columns = {len(df.select_dtypes(include=numerics).columns)}' ) # not too many non-numeric columns

# extract unique countries in the df
unique_values = set()
df['countries_in_family'].apply(lambda x: unique_values.update(x.strip("[]").replace("'", "").split())) 

# Create new columns for each unique value
for value in unique_values:
    # each country has a column (1 if the patent belong to the country 0 otherwise)
    df[value] = df['countries_in_family'].apply(lambda x: 1 if value in x else 0)

df = df[df.abstract.notna()].copy() # drop all samples without abstract

# encode company names
df['company_name_encoded'] = df.company_name.astype('category').cat.codes  # encode companies

# remove non-numeric columns
df_columns_dropped = df.drop(['publication_number', 'company_name', 'countries_in_family', 'publn_nr',
       'primary_cpc'], axis = 1)

# f0_ has the same value as commercialization, the other two shouldn't be used
df_columns_dropped = df_columns_dropped.drop(['f0_', 'centrality', 'similarity'], axis = 1)

# remove text as I can't compute min and max on it
text = df_columns_dropped[['abstract', 'description_text']] # putting them aside for later
df_columns_dropped.drop(['abstract', 'description_text'], axis=1, inplace=True)


df_no_missing = df_columns_dropped.fillna(df_columns_dropped.mean()).copy()

# extracting what we'll try to predict
y = df_no_missing['commercialized']
df_no_missing.drop('commercialized', axis= 1, inplace=True)

# dropping columns where all the value are the same (min = max) they would be zero if I apply min max rescaling
min_eq_max = df_no_missing.columns[df_no_missing.min() == df_no_missing.max()].to_list()
df_clean = df_no_missing.drop(min_eq_max, axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_clean, y, test_size=0.20, random_state=42)

#rescale 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# putting text back in
df_clean[['abstract', 'description_text']] = text  


X_train, X_test, y_train, y_test = train_test_split(df_clean, y, test_size=0.20, random_state=42)

# same vectorizer applyied to training and testing

# bag of words for abstract
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust 'max_features' as needed
X_train_ab = encode_text_colum(X_train, 'abstract', vectorizer)
X_test_ab = encode_text_colum(X_test, 'abstract', vectorizer)

# bag of words for description_text
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust 'max_features' as needed
X_train_de = encode_text_colum(X_train_ab, 'description_text', vectorizer)
X_test_de = encode_text_colum(X_test_ab, 'description_text', vectorizer)