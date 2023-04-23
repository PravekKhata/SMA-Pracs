# importing libararies

import pandas as pd
import matplotlib.pyplot as plt

# reading the dataset
df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\Hashtag Analysis\ghana_nigeria_takedown_tweets.csv')
print(df)

# Grouping the data and counting frequency of each hashtag
grouped = df.groupby('tweet_client_name')['hashtags'].apply(lambda x: pd.Series([tag for tags in x.dropna() for tag in tags.split()]).value_counts())
print("\n\n")
print(grouped)

print()
for group in grouped.index.levels[0]:
    print(f'Top 5 hashtags for {group}: \n')
    print(grouped[group].head(5))
    print()

print()
client_name = 'Twitter for Android'
fig,ax = plt.subplots(figsize = (10,15))
top_hashtags = grouped[client_name].head(5)
ax.bar(top_hashtags.index, top_hashtags.values)
ax.set_title(f'Top 5 hashtags for {client_name}')
ax.set_xlabel('Hashtags')
ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()