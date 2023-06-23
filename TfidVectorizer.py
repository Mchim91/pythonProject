# Supervised machine learning algorithm
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()

tfidf = vec.fit_transform(['I like machine learning and clustering algorithms',
                           'Apples, oranges and any kind of fruits are health',
                           'Is it feasible with machine learning algorithms?',
                           'My family is happy because of the healthy fruits'])

# print(tfidf.A)

# similarity matrix
print((tfidf*tfidf.T).A)
