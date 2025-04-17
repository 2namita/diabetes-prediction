from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable:str, mappings:dict):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings).astype(int)

        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values: 
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        q1 = X.describe()[self.variable].loc['25%']
        q3 = X.describe()[self.variable].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        for i in X.index:
            if X.loc[i, self.variable] > self.upper_bound:
                X.loc[i, self.variable]= self.upper_bound
            if X.loc[i, self.variable] < self.lower_bound:
                X.loc[i, self.variable]= self.lower_bound

        return X


class SmokingOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encoding of smoking history column """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(X[[self.variables]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Do the one shot encoding transformation
        encoded_smoking = self.encoder.transform(X[[self.variables]])

        # Get the encoded feature names
        enc_smoking_features = self.encoder.get_feature_names_out()
        #print(f"Feature:\n {enc_smoking_features}") 
        # Append encoded smoking features to X
        X[enc_smoking_features] = encoded_smoking
        
        # Drop the original smoking  column
        X.drop(columns=[self.variables], inplace=True)
        return X

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.encoder = LabelEncoder()
        #self.encoder.fit(X[[self.variables]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        # Do the one shot encoding transformation
        X[self.variables] = self.encoder.fit_transform(X[self.variables])     

        return X
