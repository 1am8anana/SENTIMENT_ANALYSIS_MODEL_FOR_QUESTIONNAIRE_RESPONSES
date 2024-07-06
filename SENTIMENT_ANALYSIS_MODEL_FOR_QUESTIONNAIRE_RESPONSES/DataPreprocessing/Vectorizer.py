# DataPreprocessing/Vectorizer.py

# Import sci-kit learn library like TfidfVectorizer to handle the tokens into matrix.
# and BaseEstimator, TransformMixin that can help us to make the Pipeline later.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Create the Vectorizer class.
class Vectorizer(BaseEstimator, TransformerMixin):
    # Create the Constructor
    def __init__(self):
        # First of this process will instantiate the TfidfVectorizer to make the tokens into the vectors. 
        self.vectorizer = TfidfVectorizer(analyzer=self.custom_analyzer, lowercase=False)
        # create empty vocabulary to prepare for the next step (fit, fit_transform).
        self.vocabulary_ = {}

    # Create custom analyzer, this will help TfidfVectorizer to plit the string to the tokens.
    def custom_analyzer(self, tokens):
        return tokens.split(' ')
    
    # Thsi function will fit the sentences to the vectorizer and then create the vocabulary.
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        self.vocabulary_ = self.vectorizer.vocabulary_
        return self

    # This function will transform dthe sentences into the number representations.
    def transform(self, X):
        return self.vectorizer.transform(X)
