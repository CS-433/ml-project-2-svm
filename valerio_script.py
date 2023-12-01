# Just for mental sanity, to better figure out what I'm doing


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


def encode_text_colum(df, text_column, vectorizer): 
    """
    Encodes a text column with the given vectorizer, drops the old column (with text),
    and returns the dataframe with the encoded text.

    Args:
        df (pd.DataFrame): The dataframe.
        text_column (str): The name of the text column to be encoded.
        vectorizer: The text vectorizer.

    Returns:
        pd.DataFrame: The dataframe with the encoded text.
    """
    # Vectorize the text column
    df_vectorized = vectorizer.fit_transform(df[text_column])
    
    # Create a dataframe from the vectorized data
    df_encoded = pd.DataFrame(df_vectorized.toarray(), columns=[f"{text_column}_{i}" for i in range(df_vectorized.shape[1])])

    # Drop the original text column
    df.drop([text_column], axis=1, inplace=True)

    # Concatenate the original dataframe with the encoded text dataframe
    df = pd.concat([df, df_encoded], axis=1)

    return df

# Example usage:
# vectorizer = TfidfVectorizer(max_features=1000)
# X_train_ab = encode_text_column(X_train, 'abstract', vectorizer)
# X_test_ab = encode_text_column(X_test, 'abstract', vectorizer)
# X_train_de = encode_text_column(X_train_ab, 'description_text', vectorizer)
# X_test_de = encode_text_column(X_test_ab, 'description_text', vectorizer)

df = pd.read_csv('data/modelready_220423.csv')

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

# remove non-numeric columns
df_columns_dropped = df.drop(['publication_number', 'company_name', 'countries_in_family', 'publn_nr',
       'primary_cpc'], axis = 1)


# f0_ has the same value as commercialization, the other two shouldn't be used
df_columns_dropped = df_columns_dropped.drop(['f0_', 'centrality', 'similarity'], axis = 1)

# remove text as I can't compute min and max on it
text = df_columns_dropped[['abstract', 'description_text']] # putting them aside for later


df_columns_dropped.drop(['abstract', 'description_text'], axis=1, inplace=True)

print(f'missing values = {df_columns_dropped.isna().sum().sum()} ')# some missin values
df_no_missing = df_columns_dropped.fillna(df_columns_dropped.mean()).copy()
print(f'missing values after filling= {df_no_missing.isna().sum().sum()} ')


# extracting what we'll try to predict
y = df_no_missing['commercialized']
df_no_missing.drop('commercialized', axis= 1, inplace=True)

# dropping columns where all the value are the same (min = max) they would be zero if I apply min max rescaling
min_eq_max = df_no_missing.columns[df_no_missing.min() == df_no_missing.max()].to_list()
df_clean = df_no_missing.drop(min_eq_max, axis=1)

# putting text back in
df_clean[['abstract', 'description_text']] = text 

# split the data
X_train, X_test, y_train, y_test = train_test_split(df_clean, y, test_size=0.20, random_state=42)

# bag of words for abstract
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust 'max_features' as needed
X_train_ab = encode_text_colum(X_train, 'abstract', vectorizer)
X_test_ab = encode_text_colum(X_test, 'abstract', vectorizer)



# Replace NaN values with zeros in X_train_ab
X_train_ab.fillna(0, inplace=True)

# Replace NaN values with zeros in X_test_ab
X_test_ab.fillna(0, inplace=True)