from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score



def encode_text_colum(df, text_column, vectorizer): 
    """
    Encodes a text column with the given vectorizer, drop the old column (with text)
    return the databased with the encoded text

    Args:
        df (pd.DataFrame): dataframe to modify
        text_column (list): column which contain text to vectorize
        vectorizer : function to vectorize

    Returns:
        pd.Dataframe : dataframe where text column has been vectorized
    """
    df_vectorized_abstract = vectorizer.fit_transform(df[text_column])
    df.drop([text_column], axis= 1, inplace=True)
    encoded_abs = pd.DataFrame(df_vectorized_abstract.toarray())
    df.reset_index(drop=True, inplace=True)
    encoded_abs.reset_index(drop=True, inplace=True)
    df = pd.concat([pd.DataFrame(df_vectorized_abstract.toarray()), df], axis=1)
    return df


def modify_df(df, cols_to_drop):
    """ Function to modify DataFrame by dropping specified columns

    Args:
        df (pd.DataFrame): dataframe from which columns will be dropped
        cols_to_drop (list):  list of columns to be dropped

    Returns:
        pd.DataFrame: dataframe with columns dropped
    """
    df_out = df.copy()
    df_out = df_out.drop(cols_to_drop, axis=1)
    return df_out


def train_RF(X_train, y_train):
    """Performs cross validation with random forest on given dataframe

    Args:
        X_train (pd.Dataframe): training set data
        y_train (pd.Dataframe): training set labels

    Returns:
        dict: disctionary with scores
    """
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_dict = cross_validate(rf_classifier, X_train, y_train, cv=5, n_jobs=5, scoring=['f1', 'accuracy', 'precision', 'recall'])       

    print(f'Average F1 Score: {np.mean(scores_dict["test_f1"])}')
    print(f'Average Accuracy: {np.mean(scores_dict["test_accuracy"])}')
    print(f'Average Precision: {np.mean(scores_dict["test_precision"])}')
    print(f'Average Recall: {np.mean(scores_dict["test_recall"])}')

    return scores_dict["test_accuracy"]


def plot_features_importance(X_train, X_test, y_train, y_test, features_to_drop): 
    """plots most important features returned from random forest

    Args:
        X_train (pd.Dataframe): training set data
        y_train (pd.Dataframe): training set labels
        X_train (pd.Dataframe): test set data
        y_train (pd.Dataframe): test set labels
        features_to_drop (list): features to be excluded
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_train)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(scaled_data, y_train)

    x_test_mod = modify_df(X_test, features_to_drop)
    y_pred = rf_classifier.predict(scaler.transform(x_test_mod))
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    sorted_data = sorted(zip(X_train.columns, rf_classifier.feature_importances_), key=lambda x: x[1], reverse = True)

    # Unzip the sorted data
    names, feature_importances = zip(*sorted_data)

    # Print the sorted lists

    plt.bar(names[:30], feature_importances[:30])
    plt.xticks(rotation = 90)
    plt.xlabel('Features')
    plt.ylabel("Feature's importance")
    plt.show()