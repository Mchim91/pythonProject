# Un-Supervised machine learning algorithm
import collections
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# stop words are common words which does not add much meaning to a sentence
# such as ['I', 'am', 'he', 'she', 'is', 'on', 'in']
nltk.download('punkt')


def tokenizer(text):
    # transform the text into an array of words
    tokens = word_tokenize(text)
    # yields the stem (fishing-fish, fisher-fish)
    stemmer = PorterStemmer()
    # we filter out stopwords
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def cluster_sentences(texts, n=2):
    # create TF-IDF again: stopwords-> we filter out common words (I,my, the, and...)
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords.words('english'), lowercase=True)
    # builds a TF-IDF matrix for the sentences
    matrix = vectorizer.fit_transform(texts)
    # fitting the k-means clustering model
    model = KMeans(n_clusters=n)
    model.fit(matrix)
    topics = collections.defaultdict(list)

    for index, label in enumerate(model.labels_):
        topics[label].append(index)

    return dict(topics)


if __name__ == '__main__':

    sentences = ["FOREX is the stock market for trading currencies",
                 "Quantum physics is quite important in science nowadays.",
                 "Investing in stocks and trading with them are not that easy",
                 "Software engineering is hotter and hotter topic in the silicon valley",
                 "Warren Buffet is famous for making good investments. He knows stock markets"]

    n_clusters = 2
    clusters = cluster_sentences(sentences, n_clusters)

    for cluster in range(n_clusters):
        print("CLUSTER ", cluster, ":")
        for i, sentence in enumerate(clusters[cluster]):
            print("\tSENTENCE ", i+1, ": ", sentences[sentence])
