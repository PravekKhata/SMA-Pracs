import ast
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community as comm
import pyvis.network as net

df = pd.read_csv('D:\Desktop\SMA\Practicals\SMA Pycharm Pracs\Social Network Analysis\preprocessed_tweets.csv')
print(df)
print()

# Taking a subset of the dataset
df = df[df['Country'].notnull()]
df.reset_index(inplace = True)
df = df[0:20]
print(df.head())

# Seperating list of hashtags

# Converting list of hashtags into a
for index,row in df.iterrows():
    df['hashtags'][index] = ast.literal_eval(row['hashtags'])

# making each row have 1 hashtag
df = df.explode('hashtags').drop_duplicates()
print(df)


countries = list(df['Country'].unique())
cities = list(df['City'].unique())
hashtags = list(df['hashtags'].unique())

# Creating a graph
G = nx.Graph()

# Adding nodes
G.add_nodes_from(cities + countries + hashtags)

# adding edges from hashtags to countries

ht_country = df.groupby(['hashtags','Country'])
for name,grp in ht_country:
    hashtag,country = name
    weight = len(grp)
    G.add_edge(hashtag,country, weight=weight,title=weight,label="Tweet: "+str(weight))

# adding edges from countries to cities
country_city = df.groupby(['Country','City'])
for name,grp in country_city:
    city,country = name
    weight = len(grp)
    G.add_edge(country,city, weight=weight,title=weight,label="Tweet: "+str(weight))

nx.draw(G,with_labels=True)
plt.axis('off')
plt.show()

# Performing community detection
print('\n\n')
community_detect = comm.girvan_newman(G)
community = next(community_detect)
# print communities
for i, community in enumerate(list(community)):
    print(f'Community {i+1}: {community}')

# Influential node detection
print('\n\n')
degree_centrality = nx.degree_centrality(G)

# printing degree centrality of each node
for node in degree_centrality:
    print(f'Node: {node}   Degree Centrality: {degree_centrality[node]}\n')

# Printing the degree centrality of most influential nodes
print('\n\n')
sorted_nodes = sorted(degree_centrality, key=degree_centrality.get,reverse=True)

for i in range(3):
    print(f'{i+1}. {sorted_nodes[i]}: {degree_centrality[sorted_nodes[i]]}')

print('\n\n')
# Compute betweenness centrality for each node
betweenness_centrality = nx.betweenness_centrality(G)

# Sort nodes by betweenness centrality in descending order
sorted_node = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)

# Print the top 10 most influential hashtags
top_hashtags = sorted_node[:10]
for hashtag in top_hashtags:
    print(hashtag)




