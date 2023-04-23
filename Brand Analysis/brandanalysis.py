import spacy
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# reading the dataset
df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\Brand Analysis\_apple_iphone_11_reviews.csv')
print()
print(df)
print()

df = df.drop(['index'],axis = 1)
print(df)
print()

# using spacy to preprocess text
sp = spacy.load('en_core_web_sm')
def preprocess_text(txt):

    doc = sp(str(txt).lower())
    tokens = [token for token in doc]

    # removing punctuations,stop words and digits from the doc
    tokens = [token for token in doc if not token.is_stop and not token.is_digit and not token.is_punct]

    # finding the lemma of the tokens
    tokens = [token.lemma_ for token in tokens]

    return ' '.join(tokens)

df['Clean Text'] = df['review_text'].apply(preprocess_text)
print()
print(df['Clean Text'])
print()

# Performing sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['Sentiment Polarity'] = df['Clean Text'].apply(get_sentiment)
print()
print(df['Sentiment Polarity'].head(20))

# Visualizing the data
colors = ['blue','red']
plt.hist(df['Sentiment Polarity'],bins=10)
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis for Apple Iphone 11 Reviews')
plt.show()