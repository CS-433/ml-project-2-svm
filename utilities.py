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
