import pandas as pd
import matplotlib.pyplot as plt

# reading and printing the dataset
df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\Location Analysis\sample.csv')
print('\n',df['Tweet Location'])

# dropping the rows that have NaN Values
df = df[df['Tweet Location'].notnull()]
print()
print(df['Tweet Location'])

# Adding the locations to a list
location_list = []
for location in df['Tweet Location']:
    location_list.append(location)

# Creating a frequency districution of the location data
location_frequency = pd.Series(location_list).value_counts()

# Printing the 10 most common location
print('\n10 Most Common Locations:\n\n', location_frequency.head(10))

# Visualizing the frequency distribution using bar chart

plt.figure(figsize=(10,6))
location_frequency.head(20).plot(kind = 'bar')

plt.title('Top 20 Most Common Locations')
plt.xlabel('Location')
plt.ylabel('Frequency')
plt.show()
