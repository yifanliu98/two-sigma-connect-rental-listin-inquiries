###########################################


#get the basic info of the train database


###########################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


color = sns.color_palette()

#matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'

train_df = pd.read_json("train.json.zip", convert_dates=['created'])
train_df.reset_index(inplace=True, drop=True)

train_df.head()

########################################


#get the basic info of the test database
#get the rows of train file



########################################

train_df.info()

###########################################


#histogram of latitude without outlier for part 1 

#histogram of latitude with outlier for part 2


###########################################

llimit = np.percentile(train_df.latitude.values, 1)
ulimit = np.percentile(train_df.latitude.values, 99)
train_df['latitude'].loc[train_df['latitude']<llimit] = llimit
train_df['latitude'].loc[train_df['latitude']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(train_df.latitude.values, bins=50, kde=True)
plt.xlabel('latitude', fontsize=12)
plt.title('Histogram of latitude without outlier')
plt.show()


#with outlier 
latitude_srs = train_df['latitude'].value_counts()
plt.figure(figsize=(8,4))
# sns.barplot(latitude_srs.index, latitude_srs.values, alpha=0.8, color=color[1])
plt.hist(latitude_srs.values, bins=1000,alpha=0.9)
# train_df.latitude.plot(kind='hist', bins=50000, color='c');
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('latitude', fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Histogram of latitude with outlier')
plt.show()

###########################################


#histogram of longitude without outlier for part 1 

#histogram of longitude with outlier for part 2


###########################################

# longitude_srs = test_df['longitude'].value_counts()

llimit = np.percentile(train_df.longitude.values, 1)
ulimit = np.percentile(train_df.longitude.values, 99)
train_df['longitude'].loc[train_df['longitude']<llimit] = llimit
train_df['longitude'].loc[train_df['longitude']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(train_df.longitude.values, bins=50, kde=True)
plt.xlabel('longitude', fontsize=12)
plt.title('Histogram of longitude without outlier')
plt.show()

#with outlier 
longitude_srs = train_df['longitude'].value_counts()
plt.figure(figsize=(8,4))
plt.hist(longitude_srs.values, bins=10000,  alpha=0.9)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('longitude', fontsize=12)
plt.title('Histogram of longitude with outlier')
plt.xticks(rotation='vertical')
plt.show()

###########################################


#histogram of price without outlier for part 1 

#histogram of price with outlier for part 2


###########################################

ulimit = np.percentile(train_df.price.values, 99)
train_df['price'].loc[train_df['price']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(train_df.price.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.title('Histogram of price without outlier')
plt.show()

#with outlier
price_srs = train_df['price'].value_counts()
plt.figure(figsize=(8,4))
plt.hist(price_srs.values, bins=10000,  alpha=0.9)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('price', fontsize=12)
plt.title('Histogram of price with outlier')
plt.xticks(rotation='vertical')
plt.show()

###########################################


#hour-wise listing for part 1



###########################################

#hour-wise listing trend and find out the top 5 busiest hours of postings
train_df["hour_created"] = train_df["created"].dt.hour
cnt_srs = train_df['hour_created'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Hour')
plt.ylabel('Counts')
plt.title('Number of Posting per Hour')
plt.show()

###########################################


#proportion of target variable values



###########################################

#show the proportion of target variable values

int_level = train_df['interest_level'].value_counts()
# print(int_level)
plt.figure(figsize=(8,4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Interest level', fontsize=12)
plt.show()

#########################################


#Find out the number of missing values in each variable.


#########################################

#find out the missing value in every variable
train_df.isnull().sum()

#outlier of latitude
train_df.latitude.plot(kind='box');
# train_df.loc[train_df.latitude == train_df.latitude.max()]

#outlier of longitude
train_df.longitude.plot(kind='box');
# train_df.loc[train_df.longitude == train_df.longitude.max()]

#outlier of price
train_df.price.plot(kind='box');
# train_df.loc[train_df.price == train_df.price.max()]

cnt_srs = train_df['bathrooms'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of bathrooms', fontsize=12)
plt.title('Histogram of bathrooms')
plt.show()