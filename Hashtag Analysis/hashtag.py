# importing libararies

import pandas as pd
import matplotlib.pyplot as plt

# reading the dataset
df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\Hashtag Analysis\ghana_nigeria_takedown_tweets.csv')
print(df)

# seperating the list of the hashtags in the dataset
df['hashtags'] = df['hashtags'].apply(lambda x: x.strip("[]").replace("'","").split(",") if len(x) >= 1 else [])
df = df.explode('hashtags')

print(df['hashtags'].unique())

# Use value_counts to count occurence of each hashtag
hashtag_counts = df['hashtags'].value_counts()
print()

# Removing the empty hashtags
hashtag_counts = hashtag_counts[hashtag_counts.index != '']
print(hashtag_counts)

# Plotting Wordcloud
top_hashtags = hashtag_counts[1:21]
wordcloud = WordCloud(width=800,height=400,background_color='white',max_words=20).generate_from_frequencies(top_hashtags)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Top 20 Hashtags')
plt.show()

# Plotting pie chart
plt.pie(top_hashtags,labels=top_hashtags.index,autopct='%1.1f%%')
plt.title('Top 20 hashtags')
plt.show()
