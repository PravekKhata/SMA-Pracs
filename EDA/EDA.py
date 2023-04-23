import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the dataset
df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\_animes.csv')
print(df.head(5))
print()

# Check Shape
print(df.shape)
print()

# Check each data type of columns and missing values using info()
print(df.info())
print()

# Summary Statistics
print('\n',df.describe())

# Checking percentage of missing values
print()
print(df.isnull().sum() / df.shape[0] * 100)

df = df.drop(['uid'],axis=1)
print(df)
print()

#visualizing the data
anime_df = df.head(20)

sns.countplot(data = anime_df, x= 'episodes')
plt.show()

# plotting a scatter plot

plt.scatter(anime_df['score'], anime_df['popularity'])
plt.xlabel('Score')
plt.ylabel('Popularity')
plt.title(' Score Vs Popularity')
plt.show()

# plotting a bar chart
plt.bar(anime_df['title'], anime_df['popularity'])
plt.xticks(rotation = 90)
plt.show()


