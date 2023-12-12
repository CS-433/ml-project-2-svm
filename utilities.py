import pandas as pd

def encode_text_colum(df, text_column, vectorizer): 
    """encodes a text column with the given vectorizer, drop the old column (with text)
    return the databased with the encoded text

    Args:
        df (_type_): _description_
        text_column (_type_): _description_
        vectorizer (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_vectorized_abstract = vectorizer.fit_transform(df[text_column])
    df.drop([text_column], axis= 1, inplace=True)
    encoded_abs = pd.DataFrame(df_vectorized_abstract.toarray())
    df.reset_index(drop=True, inplace=True)
    encoded_abs.reset_index(drop=True, inplace=True)
    df = pd.concat([pd.DataFrame(df_vectorized_abstract.toarray()), df], axis=1)
    return df


def modify_df(df, cols_to_drop):
    df_out = df.copy()
    df_out = df_out.drop(cols_to_drop, axis=1)
    return df_out

def train_RF(df, y):
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.20)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'F1_score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')