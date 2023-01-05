# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:45:25 2022

@author: vaishnav
"""

#=-----------------------------------------------------------------------------=======================================================================================================================================================================
#importing the data

import pandas as pd
df = pd.read_csv(r"C:\anaconda\New folder (2)\book_new.csv")

# Data display customization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)

df

df.info()

df.describe()

# most popular items
count = df.loc[:,:].sum()
count

for i in df.columns:
    print(df[i].value_counts())
    print()



# Top 10 Popular items
count.sort_values(0, ascending = False, inplace=True)
count = count.to_frame().reset_index()
count = count.rename(columns = {'index': 'items',0: 'count'})
count

#==================================================================================================================================================================================================================================================================================================
import matplotlib.pypplot as plt
import seaborn as sns

# Data Visualization
plt.figure(figsize = (12,8))
plt.pie(df.sum(),
       labels=df.columns,
       explode = [0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Books Purchase Rate", fontsize = 18, fontweight = 'bold')
plt.show()



plt.figure(figsize = (11,6))
ax = sns.barplot(x = 'count', y = 'items', data= count)
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 45, fontsize = 12)
plt.title('Books Purchase Frequency',  fontsize = 18, fontweight = 'bold')
for i in ax.containers:
    ax.bar_label(i,)


#==================================================================================================================================================================================================================================================================================================
from mlxtend.frequent_patterns import association_rules , apriori
from mlxtend.preprocessing import transactionencoder


# With 10% Support
freq_item=apriori(df,min_support=0.1,use_colnames=True)
freq_item


# with 70% confidence
rules=association_rules(freq_item,metric='lift',min_threshold=0.7)
rules



# A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]
rules.sort_values('lift',ascending=False)

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]


# visualization of obtained rule
plt.figure(figsize = (8,4))
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


#=================================================================================================================================================================================================================================================================


# Association rules with 5% Support and 80% confidence
# With 5% Support
freq_item1=apriori(df,min_support=0.05,use_colnames=True)
freq_item1


# With 80% confidence
rules1=association_rules(freq_item1,metric='lift',min_threshold=0.8)
rules1


rules1[rules1.lift>1]


# visualization of obtained rule
plt.figure(figsize = (8,4))
plt.scatter(rules1['support'],rules1['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


#===================================================================================================================================================================================================================

#Association rules with 20% Support and 60% confidence
# With 20% Support
freq_item2=apriori(df,min_support=0.20,use_colnames=True)
freq_item2


# With 80% confidence
rules2=association_rules(freq_item2,metric='lift',min_threshold=0.8)
rules2


rules2[rules2.lift>1]


9
