# DataPreprocessing/CustomTokenizer.py

# Import PyThaiNLP library that we need to use to handle Thai language.
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
import string
import re
# This BaseEstimator and TransformerMixin will help us to make/create the Pipeline.
# some kind like a helper function/object.
from sklearn.base import BaseEstimator, TransformerMixin

# Create Tokenizer class.
class Tokenizer(BaseEstimator, TransformerMixin):
    # This block __init__ is the Constructor of this class.
    def __init__(self):
        # First of all we need to instantiate puntuation of Thai language.
        self.punctuation = string.punctuation + 'ๆ' + 'ฯ' + '_'
        # And theninstantiate stop words of Thai word/language on the set format.
        self.stopwords = set(thai_stopwords())
    
    # This is just fit function. (That requires for make an Pipeline)
    def fit(self, X, y=None):
        return self
    # Same thing above fit fucntion but this is transform function and can clean the sentence by the process and return tokens that was cleaned.
    def transform(self, X):
        cleaned_tokens = []
        # Loop through the sentences(X) that we have recieve from X argument of this function.
        for sentence in X:
            # In short this line will remove the sentence/word is not in the punctuation list.
            tokens = "".join(char for char in sentence if char not in self.punctuation)
            # Instantiate the tokenizer from PyThaiNLP. and make the tokens.
            tokens = word_tokenize(tokens, engine='newmm', keep_whitespace=False)
            # some how in our case the tokens still have punctuation because som tokens was not in the right/correct sysntax.
            # and that's it we need to handle it again to be clear.
            clean_tokens = [re.sub(rf"[{re.escape(self.punctuation)}]", '', token) for token in tokens]
            # And these line of code will reduces/removes some weird tokens like 'กันนน' or 'น' etc.
            clean_tokens = [re.sub(r'(\w)\1{2,}', r'\1', token) for token in clean_tokens]
            clean_tokens = [token for token in clean_tokens if token and len(token) > 1 and not token.isdigit()]
            # Remove stop word from dthe tokens.
            clean_tokens = [token for token in clean_tokens if token not in self.stopwords]
            # Now this process will conects/combines many tokens individual sentence/row to string.
            clean_tokens_str = " ".join(token for token in clean_tokens)
            cleaned_tokens.append(clean_tokens_str)
        return cleaned_tokens