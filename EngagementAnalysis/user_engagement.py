import pandas as pd
import matplotlib.pyplot as plt

# reading the dataset
df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\EngagementAnalysis\MrBeast.csv')

print(df)

# Sorting values
likes_df = df.sort_values(by='likeCount',ascending=False).head(10)
print()
print(likes_df)

# Plotting the bar chart
plt.bar(likes_df['content'],likes_df['likeCount'])

plt.title('Likes by Tweet Content')
plt.xlabel('Tweet Content')
plt.ylabel('Number of Likes')
plt.xticks(rotation = 90)
plt.show()