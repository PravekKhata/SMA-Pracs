# importing libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis
from pyLDAvis.gensim import prepare

# reading the csv file

df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\sma_dataset.csv')
print(df)
print()


stop_words = set(stopwords.words('english'))

# Tokenize tweets
tweets = []
for tweet in df['Tweet']:
    tokens = []
    for word in word_tokenize(tweet.lower()):
        if word not in stop_words and not word.startswith('@') and not word.startswith('http'):
            tokens.append(word)
    tweets.append(tokens)


# Creating a dictionary
dictionary = Dictionary(tweets)
print(dictionary)
print()

# Converting document into bag of words using doc2bow()
corpus = [dictionary.doc2bow(tweet) for tweet in tweets]
print(corpus)
print()


# Training the LDA model
# passes: the number of times the entire corpus should be iterated over during training
# id2word: the dictionary that maps word indices to words.
# num_topics: the number of topics to discover in the corpus
lda_model = LdaModel(corpus=corpus,num_topics=10,id2word=dictionary,passes=10)
print("\n\n")


# Printing the learned topic
for topic in lda_model.show_topics():
    print(topic,"\n")

# Extra: Computing Perplexity using log_perplexity function

print("\nPerplexity: ", lda_model.log_perplexity(corpus))

# # visualizing the topic model
#
# pyLDAvis.enable_notebook()
#
# vis = prepare(lda_model,corpus,dictionary=dictionary)
# print(vis)




