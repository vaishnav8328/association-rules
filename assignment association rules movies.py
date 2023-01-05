# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:59:16 2022

@author: vaishnav
"""

#========================================================================================================================================================================================================================================================================
#importing the data

import pandas as pd
df = pd.read_csv(r"C:\anaconda\New folder (2)\my_movies.csv")

# Data display customization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)

df


df1 = df.iloc[:,5:]
df1

#========================================================================================================================================================================================================================================================================

df1.info()

df1.describe()

# most popular items
count = df1.loc[:,:].sum()
count


for i in df1.columns:
    print(df1[i].value_counts())
    print()


print('The unique books sold are {}'.format(df1.columns))
print("The number of unique books sold are {} ".format(df1.columns.size))


# Top 10 Popular items
count.sort_values(0, ascending = False, inplace=True)
count = count.to_frame().reset_index()
count = count.rename(columns = {'index': 'items',0: 'count'})
count


#========================================================================================================================================================================================================================================================================
#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = (15,12)
wordcloud = WordCloud(background_color = 'black', width = 1200,  height = 1200, max_words = 121).generate(str(df1.sum()))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items',fontsize = 20)
plt.show()




plt.figure(figsize = (12,8))
plt.pie(df1.sum(),
       labels=df1.columns,
       explode = [0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Movies with respect to Purchase Rate", fontsize = 18, fontweight = 'bold')
plt.show()



plt.figure(figsize = (11,6))
ax = sns.barplot(x = 'count', y = 'items', data= count)
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 45, fontsize = 12)
plt.title('Books Purchase Frequency',  fontsize = 18, fontweight = 'bold')
for i in ax.containers:
    ax.bar_label(i,)


#========================================================================================================================================================================================================================================================================
from mlxtend.frequent_patterns import association_rules , apriori
from mlxtend.preprocessing import transactionencoder

#Association rules with 10% Support and 70% confidence

# with 10% support
freq_item=apriori(df1,min_support=0.1,use_colnames=True)
freq_item



# 70% confidence
rules=association_rules(freq_item,metric='lift',min_threshold=0.7)
rules


# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]


# visualization of obtained rule
plt.figure(figsize = (8,4))
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


#========================================================================================================================================================================================================================================================================

#Association rules with 5% Support and 90% confidence
# with 5% support
freq_item1=apriori(df1,min_support=0.05,use_colnames=True)
freq_item1


# 90% confidence
rules1=association_rules(freq_item1,metric='lift',min_threshold=0.9)
rules1


rules1[rules1.lift>1]


# visualization of obtained rule
plt.figure(figsize = (8,4))
plt.scatter(rules1['support'],rules1['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

#========================================================================================================================================================================================================================================================================







