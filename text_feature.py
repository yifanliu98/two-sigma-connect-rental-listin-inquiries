import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_df = pd.read_json("train.json.zip", convert_dates=['created'])
# train_df = train_df[train_df['building_id'] != '0']
train_df.reset_index(inplace=True, drop=True)

######################################################################################
# dealing with missing value on building_id
######################################################################################


# from math import cos, asin, sqrt
# # https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
# def distance_pair(lat1, lon1, lat2, lon2):
#     p = 0.017453292519943295     #Pi/180
#     a = 0.5 - cos((lat2 - lat1) * p)/2 +  cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
#     d_2_point = 6371 *2 * asin(sqrt(a)) #2*R*asin...
#     return d_2_point
# distance_pairs = np.vectorize(distance_pair)

# def get_id(row):
#     id_list = train_df[['building_id', 'display_address', 'latitude', 'longitude']]
#     id_list = id_list[id_list['display_address'] == row['display_address']]
    
#     if id_list.shape[0] == 0:
#         return '0'
#     del id_list['display_address']
#     id_list = id_list[id_list['building_id'] != '0']
#     id_list = id_list.groupby(['building_id']).agg(lambda x: pd.Series.mode(x)[0]).reset_index()
#     id_list['distance'] = distance_pairs(id_list['latitude'], id_list['longitude'], row['latitude'], row['longitude'])
#     index = id_list['distance'].idxmin()
#     return id_list['building_id'][index]

# # building without building_id
# building_no_id = train_df[train_df['building_id'] == '0'].reset_index(drop=True)
# building_no_id['id'] = building_no_id.index
# building_no_id['building_id'] = building_no_id.apply(lambda row: get_id(row), axis=1)

# train_df = np.concat([building_no_id, train_df[train_df['building_id'] != '0']], axis=1)


######################################################################################


created_date = train_df[['building_id', 'created']]
created_date = created_date[created_date['building_id'] != '0']
created_date = created_date.groupby('building_id').max().reset_index()

# groupby all building with same building_id based on the latest posting
building = pd.merge(train_df, created_date, on=['building_id', 'created'], how='inner')

# finding all features
features_dict = {}
def get_features(features):
    for f in features:
        f = f.strip('!')
        f = f.strip('*')
        f = f.strip(' ')
        
        if f in features_dict.keys():
            features_dict[f] += 1
        else:
            features_dict[f] = 1

a = building['features'].apply(get_features)

# print all features and number of occurrences
# print(features_dict)

feature_columns = {}
feature_list = []
for key in features_dict.keys():
    if features_dict[key] >= 800:
        feature_list.append(key)
        feature_columns[key] = []

def get_feature_columns(features):
    for key in feature_columns:
        if key in features:
            feature_columns[key].append(True)
        else:
            feature_columns[key].append(False)
            
temp = building['features'].apply(get_feature_columns)
building = pd.concat([building, pd.DataFrame.from_dict(feature_columns)], axis=1)

building.head()



