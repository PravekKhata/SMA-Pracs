import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk import PorterStemmer,word_tokenize
from nltk.corpus import stopwords
import spacy

df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\sma_dataset.csv')
print(df)

nlp = spacy.load('en_core_web_sm')
# preprocessing
def preprocess(text):

    # text = text.lower()
    #
    # # Removing punctutions
    # text = text.translate(str.maketrans('','',string.punctuation))
    #
    # # Tokenizing words and Removing stopwords
    # words = nltk.word_tokenize(text)
    # stop_words = set(stopwords.words(('english')))
    # words = [word for word in words if word not in stop_words]
    #
    # # Stem the words
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words]
    doc = nlp(str(text).lower())

    tokens = [token for token in doc]
    tokens = [token for token in doc if not token.is_stop and not token.is_punct and not token.is_digit]
    print(tokens)
    print()

    # finding lemma
    tokens = [token.lemma_ for token in tokens]

    # Join the words back into a string
    text = ' '.join(tokens)

    return text

# Cleaning/ Preprocessing the tweets
df['Clean Tweet'] = df['Tweet'].apply(preprocess)

# Getting the polarity from the tweets
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['Sentiment_polarity'] = df['Clean Tweet'].apply(get_sentiment)

sentiment = df[['Clean Tweet','Sentiment_polarity']]
print()
print(sentiment)

# identifying the tweet as positive,negative or neutral
pos = 0
neg = 0
neu = 0

sentiment_list = []

# Create a loop to classify the tweets as Positive, Negative, or Neutral.
# Count the number of each

for items in df['Sentiment_polarity']:

    if items > 0:
        s = 'Positive'
        pos = pos + 1
        sentiment_list.append(s)

    elif items < 0:
        s = 'Negative'
        neg = neg + 1
        sentiment_list.append(s)

    else:
        s = 'Neutral'
        neu = neu + 1
        sentiment_list.append(s)

df['Sentiment'] = sentiment_list
print("\n\n",df[['Clean Tweet','Sentiment']].head(5))

sentiment_counts = df['Sentiment'].value_counts()

# plotting a bar graph
plt.bar(sentiment_counts.index,sentiment_counts.values)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Plotting a pie graph
plt.pie(sentiment_counts,autopct='%1.1f%%',startangle=90,labels=sentiment_counts.index)
plt.title('Sentiment Distribution')
plt.show()
