import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

np.random.seed(416)

# Setup
text = pd.read_csv('tweets-2020-4-30.csv')
text = text.fillna('') # some rows are nan so replace with empty string
vectorizer = TfidfVectorizer(max_df=0.95)
tf_idf = vectorizer.fit_transform(text['text'])
feature_names = vectorizer.get_feature_names()


num_tweets = tf_idf.shape[0]
num_words = tf_idf.shape[1]

nmf = NMF(n_components=5, init='nndsvd')
tweets_projected = nmf.fit_transform(tf_idf)

q3 = 'word'


small_words = ['dogs', 'cats', 'axolotl']
small_weights = np.array([1, 4, 2])


small_words_argsort = np.argsort(small_weights)[::-1]
sorted_small_words = [small_words[i] for i in small_words_argsort]


def words_from_topic(topic, feature_names):
    """
    Sorts the words by their weight in the given topic from largest to smallest.
    topic and feature_names should have the same number of entries.

    Args:
     - topic (np.array): A numpy array with one entry per word that shows the weight in this topic.
    - feature_names (list): A list of words that each entry in topic corresponds to

    Returns:
    - A list of words in feature_names sorted by weight in topic from largest to smallest.
    """
    topic_argsort = np.argsort(topic)[::-1]
    sorted_feature_name = [feature_names[i] for i in topic_argsort]
    return sorted_feature_name


q6 = 2

rows = np.argmax(tweets_projected, axis=1)
columns = np.bincount(rows)

largest_topic = np.argmax(columns)

nmf_small = NMF(n_components=3, init='nndsvd')
tweets_projected_small = nmf_small.fit_transform(tf_idf)

second_topic = tweets_projected_small[:, 2]
result = np.where(second_topic >= 0.15)
index_second_topic = text.iloc[result]['text']
outlier_tweets = np.unique(index_second_topic)
