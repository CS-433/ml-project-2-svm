from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier




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

def train_RF(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_dict = cross_validate(rf_classifier, X_train, y_train, cv=5, n_jobs=5, scoring=['f1', 'accuracy', 'precision', 'recall'])       

    print(f'Average F1 Score: {np.mean(scores_dict["test_f1"])}')
    print(f'Average Accuracy: {np.mean(scores_dict["test_accuracy"])}')
    print(f'Average Precision: {np.mean(scores_dict["test_precision"])}')
    print(f'Average Recall: {np.mean(scores_dict["test_recall"])}')

    return scores_dict["test_accuracy"]







