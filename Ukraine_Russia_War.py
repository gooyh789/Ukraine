# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:16:47 2022

@author: gooyh
"""
#import modules

import pandas as pd
import numpy as np

# import the data from github
csv_url = "https://raw.githubusercontent.com/zhukovyuri/VIINA/master/Data/control_latest.csv"
control = pd.read_csv(csv_url, error_bad_lines=False)
control = control.drop_duplicates(['longitude', 'latitude'])
control = control.drop_duplicates(['name', 'latitude'])
control = control.dropna()
control = control.rename(columns = {'name':'city'})


world = pd.read_csv('C:\\Users\\gooyh\\Documents\\Python\\Ukrain_Russia_War\\worldcities.csv') # world cities database (https://simplemaps.com/data/world-cities)
ukraine = world.loc[world['iso2'] == 'UA']
population = ukraine[['city', 'lat', 'lng','population', 'admin_name']]
population = population.reset_index()
population = population[['city', 'lat', 'lng','population', 'admin_name']]
population = population.rename(columns = {'lat':'latitude','lng':'longitude'})

language = pd.read_csv('C:\\Users\\gooyh\\Documents\\Python\\Ukrain_Russia_War\\ua_lang.csv') # https://translatorswithoutborders.org/language-data-for-ukraine
lang = language[['admin1_name','admin2_name', 'Ukrainian L1', 'Russian L1']]
lang = lang.drop_duplicates(['admin1_name','admin2_name'])
lang = lang.reset_index()
lang = lang[['admin1_name', 'admin2_name', 'Ukrainian L1', 'Russian L1']]
lang = lang.rename(columns={'admin1_name':'admin_name', 'admin2_name':'city'})


population['admin_name'] = population['admin_name'].replace({'Kyyivs':'Kyivska oblast', 'Vinnyts':'Vinnytska oblast',
                                                             'Zhytomyrs':'Zhytomyrska oblast', 'Chernihivs':'Chernihivska oblast',
                                                             'Cherkas':'Cherkaska oblast', 'Sums':'Sumska oblast',
                                                             'Poltavs':'Poltavska oblast', 'Kirovohrads':'Kirovohradska oblast',
                                                             'Kyyiv':'Kyiv (independent)', 'Donets':'Donetska oblast',
                                                             'Luhans':'Luhanska oblast', 'Kharkivs':'Kharkivska oblast',
                                                             'Zaporiz':'Zaporizka oblast', 'Odes':'Odeska oblast',
                                                             'Krym':'The Autonomous Republic of Crimea', 'Khersons':'Khersonska oblast',
                                                             'Mykolayivs':'Mykolaivska oblast', 'Sevastopol':'Sevastopol (independent)',
                                                             'Dnipropetrovs':'Dnipropetrovska oblast', 'Lvivs':'Lvivska oblast',
                                                             'Zakarpats':'Zakarpatska oblast', 'Ivano-Frankivs':'Ivano-Frankivska oblast',
                                                             'Volyns':'Volynska oblast', 'Khmel':'Khmelnytska oblast',
                                                             'Rivnens':'Rivnenska oblast', 'Ternopils':'Ternopilska oblast',
                                                             'Chernivets':'Chernivetska oblast'})

new_set = population.merge(lang, on = ['city', 'admin_name'], how = 'left')
new_set.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\new.csv", index=False)

center = ['Kyivska oblast', 'Vinnytska oblast', 'Zhytomyrska oblast', 'Chernihivska oblast',
          'Cherkaska oblast', 'Sumska oblast', 'Poltavska oblast', 'Kirovohradska oblast', 'Kyiv (independent)']
east = ['Donetska oblast', 'Luhanska oblast', 'Kharkivska oblast']
south = ['Zaporizka oblast', 'Odeska oblast', 'The Autonomous Republic of Crimea', 'Khersonska oblast',
         'Mykolaivska oblast', 'Sevastopol (independent)', 'Dnipropetrovska oblast']
west = ['Lvivska oblast', 'Zakarpatska oblast', 'Ivano-Frankivska oblast', 'Volynska oblast', 
        'Khmelnytska oblast', 'Rivnenska oblast', 'Ternopilska oblast', 'Chernivetska oblast']
region = []
for i in range(len(new_set)):
    if new_set['admin_name'][i] in center:
        region.append('center')
    elif new_set['admin_name'][i] in east:
        region.append('east')
    elif new_set['admin_name'][i] in south:
        region.append('south')
    elif new_set['admin_name'][i] in west:
        region.append('west')

new_set['region'] = region
new_set.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\test.csv", index=False)


# data manipulation: Simplify the date, Select the city, where "CONTESTED" mark exists.
new = control.drop(['asciiname', 'alternatenames'],1)
new.columns = new.columns.str.replace('ctr_', '')
dt_list = ['geonameid', 'city', 'longitude', 'latitude', 'feature_code']
for i in range(5, (len(new.columns))):
    dt_str = new.columns[i][:8]
    dt_list.append(dt_str)

new.columns = dt_list

# data: Numpy (After date simplification, CONTESTED selection)
new1 = new.to_numpy()
data = np.ndarray(shape=(0,new1.shape[1]))
for i in range(len(new1)):
    if "CONTESTED" in new1[i]:
        renew = new1[i].reshape(1,new1[i].shape[0])
        data = np.append(data, renew, axis=0)
        
# data: Unify the columns if they have same date
# If one of values among colunms is "CONTESTED", just write the "CONTESTED"
dateInfo = set(new.columns[5:])
data = pd.DataFrame(data, columns = new.columns)
warDate = {}
for i in range(data.shape[0]):
    for date in dateInfo:
        if len(list(data.iloc[i][date][0])) == 1:
                warDate[data.iloc[i]['geonameid'], date] = data.iloc[i][date]
        else:
            if "CONTESTED" in list(data.iloc[i][date]):
                warDate[data.iloc[i]['geonameid'], date] = "CONTESTED"
            else:
                warDate[data.iloc[i]['geonameid'], date] = data.iloc[i][date][0]
    print(f"processing...{i/data.shape[0]}")
    
df = pd.Series(warDate).reset_index()
df.columns = ['geonameid', 'date', 'country']
df2 = df.pivot(index = "geonameid", columns = "date", values = 'country')
df2 = df2.reset_index()
df1 = data.iloc[:,:5]
df_combine = df1.merge(df2, how='left', on='geonameid')
warData_columns = df_combine.columns

# Date divide 
df21 = df2.loc[:,:'20220325']
df22 = df2.loc[:,'20220325':]
df22['geonameid'] = df21['geonameid']

df_combine1 = df1.merge(df21, how='left', on='geonameid')
df_combine2 = df1.merge(df22, how='left', on='geonameid')

df_combine1_columns = df_combine1.columns
df_combine2_columns = df_combine2.columns

#########################################################################################################################
# warData: Dataframe (Adding 1. how many times "CONTESTED" remained in each city, 2. if the city occupication was changed)
#########################################################################################################################

df_combine = df_combine.to_numpy() 
df_combine1 = df_combine1.to_numpy() 
df_combine2 = df_combine2.to_numpy() 

data1 = np.ndarray(shape=(0,df_combine1.shape[1]))
for i in range(len(df_combine1)):
    if "CONTESTED" in df_combine1[i]:
        renew = df_combine1[i].reshape(1,df_combine1[i].shape[0])
        data1 = np.append(data1, renew, axis=0)
df_combine1 = data1

data2 = np.ndarray(shape=(0,df_combine2.shape[1]))
for i in range(len(df_combine2)):
    if "CONTESTED" in df_combine2[i]:
        renew = df_combine2[i].reshape(1,df_combine2[i].shape[0])
        data2 = np.append(data2, renew, axis=0)
df_combine2 = data2


contested = []
for i in range(len(df_combine)):
    count = 0
    for j in range(df_combine.shape[1]):
        if df_combine[i][j] == "CONTESTED":
            count += 1
    contested.append(count)

initial = []
for i in range(len(df_combine)):
    initial.append(df_combine[i][4])

current = []
for i in range(len(df_combine)):
    current.append(df_combine[i][df_combine.shape[1]-1])

occupy_change = []
for i in range(len(df_combine)):
    con = 0
    ur = 0
    ru = 0
    for j in range(df_combine.shape[1]):
        if df_combine[i][j] == "CONTESTED":
            con = 1
        elif df_combine[i][j] == "UA":
            ur = 1
        else:
            ru = 1
    if con+ur+ru == 2:
        occupy_change.append("No")
    else:
        occupy_change.append("Yes")
        
warData = pd.DataFrame(df_combine, columns = warData_columns)
warData['initial'] = initial
warData['current'] = current
warData['occupy_change'] = occupy_change
warData['contested'] = contested

contested1 = []
for i in range(len(df_combine1)):
    count = 0
    for j in range(df_combine1.shape[1]):
        if df_combine1[i][j] == "CONTESTED":
            count += 1
    contested1.append(count)

initial1 = []
for i in range(len(df_combine1)):
    initial1.append(df_combine1[i][4])

current1 = []
for i in range(len(df_combine1)):
    current1.append(df_combine1[i][df_combine1.shape[1]-1])

occupy_change1 = []
for i in range(len(df_combine1)):
    con = 0
    ur = 0
    ru = 0
    for j in range(df_combine1.shape[1]):
        if df_combine1[i][j] == "CONTESTED":
            con = 1
        elif df_combine1[i][j] == "UA":
            ur = 1
        else:
            ru = 1
    if con+ur+ru == 2:
        occupy_change1.append("No")
    else:
        occupy_change1.append("Yes")
        
warData1 = pd.DataFrame(df_combine1, columns = df_combine1_columns)
warData1['initial'] = initial1
warData1['current'] = current1
warData1['occupy_change'] = occupy_change1
warData1['contested'] = contested1

contested2 = []
for i in range(len(df_combine2)):
    count = 0
    for j in range(df_combine2.shape[1]):
        if df_combine2[i][j] == "CONTESTED":
            count += 1
    contested2.append(count)

initial2 = []
for i in range(len(df_combine2)):
    initial2.append(df_combine2[i][4])

current2 = []
for i in range(len(df_combine2)):
    current2.append(df_combine2[i][df_combine2.shape[1]-1])

occupy_change2 = []
for i in range(len(df_combine2)):
    con = 0
    ur = 0
    ru = 0
    for j in range(df_combine2.shape[1]):
        if df_combine2[i][j] == "CONTESTED":
            con = 1
        elif df_combine2[i][j] == "UA":
            ur = 1
        else:
            ru = 1
    if con+ur+ru == 2:
        occupy_change2.append("No")
    else:
        occupy_change2.append("Yes")
        
warData2 = pd.DataFrame(df_combine2, columns = df_combine2_columns)
warData2['initial'] = initial2
warData2['current'] = current2
warData2['occupy_change'] = occupy_change2
warData2['contested'] = contested2


import os
warData.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\warData.csv", index=False)
warData1.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\warData1.csv", index=False)
warData2.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\warData2.csv", index=False)





















################ Clustering #################
# attach addtional data
war1 = warData1[['longitude', 'latitude', 'feature_code', 'current', 'contested']]

# K-prototype analysis

war1.info()
war1['longitude'] = pd.to_numeric(war1['longitude'])
war1['latitude'] = pd.to_numeric(war1['latitude'])
lgbm_data = war1.copy()

war1.select_dtypes('object').nunique()
war1.isna().sum()

from kmodes.kprototypes import KPrototypes
import plotnine
from plotnine import *
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# scale numerical columns 
for c in war1.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    war1[c] =  pt.fit_transform(np.array(war1[c]).reshape(-1, 1))

catColumnsPos = [war1.columns.get_loc(col) for col in list(war1.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(war1.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))

cl_Matrix = war1.to_numpy()

cost = []
for cluster in range(1, 10):
    try:
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(cl_Matrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break

# Converting the results into a dataframe and plotting them
cl_cost = pd.DataFrame({'Cluster':range(1, 10), 'Cost':cost})
cl_cost.head()

plotnine.options.figure_size = (8, 4.8)
(
    ggplot(data = cl_cost)+
    geom_line(aes(x = 'Cluster',
                  y = 'Cost'))+
    geom_point(aes(x = 'Cluster',
                   y = 'Cost'))+
    geom_label(aes(x = 'Cluster',
                   y = 'Cost',
                   label = 'Cluster'),
               size = 10,
               nudge_y = 1000) +
    labs(title = 'Optimal number of cluster with Elbow Method')+
    xlab('Number of Clusters k')+
    ylab('Cost')
)

# Fit the cluster
kprototype = KPrototypes(n_jobs = -1, n_clusters = 4, init = 'Huang', random_state = 0)
kprototype.fit_predict(cl_Matrix, categorical = catColumnsPos)

# Add the cluster to the dataframe
war1['cluster_id'] = kprototype.labels_
war1['segment'] = war1['cluster_id'].map({0:'Cluster 1', 1:'Cluster 2', 2:'Cluster 3', 3:'Cluster 4',
                                           4:'Cluster 5', 5:'Cluster 6', 6:'Cluster 7', 7:'Cluster 8'})

war1['longitude'] = warData1['longitude']
war1['latitude'] = warData1['latitude']
war1['contested'] = warData1['contested']
war1.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\total1.csv", index=False)




for c in lgbm_data.select_dtypes(include='object'):
    lgbm_data[c] = lgbm_data[c].astype('category')

import lightgbm as ltb
from sklearn.model_selection import cross_val_score
clf_kp = ltb.LGBMClassifier(colsample_by_tree=0.8)
cv_scores_kp = cross_val_score(clf_kp, lgbm_data, kprototype.labels_, scoring='f1_weighted')
print(f'CV F1 score for K-Prototypes clusters is {np.mean(cv_scores_kp)}')

import shap
clf_kp.fit(lgbm_data, kprototype.labels_)
explainer_kp = shap.TreeExplainer(clf_kp)
shap_values_kp = explainer_kp.shap_values(lgbm_data)
shap.summary_plot(shap_values_kp, lgbm_data, plot_type="bar", plot_size=(15, 10))






















# Before March 29
total1 = temp1
df1_cl = total1.drop(['geonameid', 'city', 'feature_code'], axis=1)
df1_cl = total1.drop(['geonameid', 'city', 'admin_name', 'feature_code'], axis=1)
# scale numerical columns 
for c in df1_cl.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    df1_cl[c] =  pt.fit_transform(np.array(df1_cl[c]).reshape(-1, 1))

catColumnsPos = [df1_cl.columns.get_loc(col) for col in list(df1_cl.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(df1_cl.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))

cl1_Matrix = df1_cl.to_numpy()

cost = []
for cluster in range(1, 10):
    try:
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(cl1_Matrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break

# Fit the cluster
kprototype = KPrototypes(n_jobs = -1, n_clusters = 2, init = 'Huang', random_state = 0)
clusters1 = kprototype.fit_predict(cl1_Matrix, categorical = catColumnsPos)


# Add the cluster to the dataframe
total1['cluster_id'] = kprototype.labels_
total1['segment'] = total1['cluster_id'].map({0:'Cluster 1', 1:'Cluster 2'})

total1.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\total1.csv", index=False)

# Cluster interpretation
cluster_interpret1 = total1.groupby('segment').agg(
    {
        'cluster_id':'count',
        'current': lambda x: x.value_counts().index[0],
        'region': lambda x: x.value_counts().index[0],
        'population': 'mean',
        'contested': 'mean',
        'Ukrainian L1': 'mean',
        'Russian L1': 'mean',
    }
).reset_index()

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x='segment', y='contested',data=total1, palette='rainbow')
plt.title('Contested day by Cluster', fontsize=17)

sns.boxplot(x='segment', y='longitude',data=total1, palette='rainbow')
plt.title('Contested day by Cluster', fontsize=17)

sns.boxplot(x='segment', y='latitude',data=total1, palette='rainbow')
plt.title('Contested day by Cluster', fontsize=17)

sns.boxplot(x='segment', y='population',data=total1, palette='rainbow')
plt.title('Contested day by Cluster', fontsize=17)

sns.boxplot(x='segment', y='Ukrainian L1',data=total1, palette='rainbow')
plt.title('Proportion of Ukrainian Language by Cluster', fontsize=17)

sns.boxplot(x='segment', y='Russian L1',data=total1, palette='rainbow')
plt.title('Proportion of Russian Language by Cluster', fontsize=17)

con1 = pd.DataFrame()
con1['prop'] = total1['contested']/29
con1['class'] = 'contested'
con1['cluster'] = total1['segment']
pop1 = pd.DataFrame()
pop1['prop'] = total1['population']/max(total1['population'])
pop1['class'] = 'population'
pop1['cluster'] = total1['segment']
ua1 = pd.DataFrame()
ua1['prop'] = total1['Ukrainian L1']
ua1['class'] = 'UkranianL1'
ua1['cluster'] = total1['segment']
ru1 = pd.DataFrame()
ru1['prop'] = total1['Russian L1']
ru1['class'] = 'RussianL1'
ru1['cluster'] = total1['segment']
box1 = pd.concat([con1, pop1, ua1, ru1])

sns.catplot(x='class', y='prop',data=box1, hue='cluster', kind='box')
plt.title('Box plot of numeric variables by clusters', fontsize=17)

cur1 = pd.DataFrame()
cur1['subject'] = total1['current']
cur1['class'] = 'current status'
cur1['cluster'] = total1['segment']
reg1 = pd.DataFrame()
reg1['subject'] = total1['region']
reg1['class'] = 'region'
reg1['cluster'] = total1['segment']
bar1 = pd.concat([cur1])
sns.catplot(x='subject',data=bar1, hue='cluster', kind='count')
plt.title('Bar plot of categorical variables by clusters', fontsize=17)

