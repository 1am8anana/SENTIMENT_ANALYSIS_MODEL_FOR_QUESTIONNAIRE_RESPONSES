# Import the needed Library/module like: Pipeline from sklearn to help us make the Pipeline easily.
# CustomTokenizer and Vectorizer to get the tokens and vector representations.
from sklearn.pipeline import Pipeline
from DataPreprocessing.CustomTokenizer import Tokenizer
from DataPreprocessing.Vectorizer import Vectorizer

# Create CustomPipeline class (easy to use in other file).
class CustomPipeline():
    # Create an Constructor of the Pipeline that will have some sequences
    # 1. Make the to tokens from Tokenizer
    # 2. transform the tokens into Vector/numeric representations.
    # 3. Fit of these data that have extracted into the Classifier.
    def __init__(self, classifier):
        self.pipeline = Pipeline([
            ('tokenizer', Tokenizer()),
            ('vectorizer', Vectorizer()),
            ('classifier', classifier)
        ])
    
    # This function will help the classifier to set the hyperparameters.
    def set_params(self, **params):
        self.pipeline.set_params(**params)
    
    # Fit function the fit the data with the Pipeline.
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    # These predict and predict_proba will get the predictions.
    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    # Thsi function will return the accuracy of the Pipeline if we need to.
    def score(self, X, y):
        return self.pipeline.score(X, y)