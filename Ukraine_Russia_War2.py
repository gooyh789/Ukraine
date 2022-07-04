# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 21:50:30 2022

@author: gooyh
"""
import pandas as pd
import numpy as np

'''
csv_url = "https://raw.githubusercontent.com/zhukovyuri/VIINA/master/Data/events_latest.csv"
event = pd.read_csv(csv_url, error_bad_lines=False)
event = event.drop_duplicates(['longitude', 'latitude', 'date'])
Event = event[['date', 'longitude', 'latitude', 'a_rus_pred','t_airstrike_pred', 't_armor_pred',
      't_raid_pred', 't_cyber_pred', 't_artillery_pred', 't_milcas_pred']]

a_list = []
for i in range(Event.shape[0]):
    if ((Event.iloc[i][1] <= 35.5) & (Event.iloc[i][2] >= 49)):
        a_list.append('center')
    else:
        a_list.append('east')
        
Event['region'] = a_list

Event['region'].mask(Event['longitude'] <= 27.5, 'west', inplace=True)
Event['region'].mask((Event['region'] == 'east') & (Event['longitude'] < 33.5) &
                     (Event['latitude'] > 48), 'center', inplace=True)
Event['region'].mask((Event['region'] == 'east') & (Event['longitude'] <= 36.8) &
                     (Event['latitude'] <= 48.8), 'south', inplace=True)

death = Event[['date', 'longitude', 'latitude', 't_milcas_pred', 'region']]
Event = Event.round({"longitude":3, "latitude":3, "a_rus_pred":0, "t_airstrike_pred":0, "t_armor_pred":0,
                     "t_raid_pred":0, "t_cyber_pred":0, 't_artillery_pred':0, 't_milcas_pred':0})

Event = Event.loc[Event['a_rus_pred'] == 1]
Event = Event.loc[((Event['t_airstrike_pred'] == 1) | (Event['t_armor_pred'] == 1) | 
                  (Event['t_raid_pred'] == 1) | (Event['t_cyber_pred'] == 1) |
                  (Event['t_artillery_pred'] == 1))]
Event.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\event.csv", index=False)
'''



Event = pd.read_csv(r"C:\Users\gooyh\Documents\R\Ukraine\event.csv")
Event['region'].value_counts().plot(kind='bar')


Event_before = Event.loc[Event['date'] <= 20220325]
#Event_before = Event.loc[Event['date'] <= 20220615]
Event_before.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\event1.csv", index=False)
Event_before['region'].value_counts().plot(kind='bar')
act = []
for i in range(Event_before.shape[0]):
    if Event_before.iloc[i][4] == 1:
        act.append('air strike')
    elif Event_before.iloc[i][5] == 1:
        act.append('armor')
    elif Event_before.iloc[i][6] == 1:
        act.append('special')
    elif Event_before.iloc[i][7] == 1:
        act.append('cyber')
    elif Event_before.iloc[i][8] == 1:
        act.append('artillery')
Event_before['type'] = act

import matplotlib.pyplot as plt
import seaborn as sns
plots = sns.countplot(data=Event_before, x='type', hue='region')
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)

east = Event_before.loc[Event_before['region'] == 'east']
center = Event_before.loc[Event_before['region'] == 'center']
west = Event_before.loc[Event_before['region'] == 'west']
south = Event_before.loc[Event_before['region'] == 'south']

act_east = east.groupby(['date']).agg({'t_airstrike_pred': ['sum'],
                                           't_armor_pred': ['sum'],
                                           't_raid_pred': ['sum'],
                                           't_cyber_pred': ['sum'],
                                           't_artillery_pred': ['sum']})
act_east.columns = act_east.columns.droplevel(1)
act_east = act_east.rename(columns = {'t_airstrike_pred':'air_strike_east',
                              't_armor_pred':'armor_east',
                              't_raid_pred':'special_east',
                              't_cyber_pred':'cyber_east',
                              't_artillery_pred':'artillery_east'})

act_center = center.groupby(['date']).agg({'t_airstrike_pred': ['sum'],
                                           't_armor_pred': ['sum'],
                                           't_raid_pred': ['sum'],
                                           't_cyber_pred': ['sum'],
                                           't_artillery_pred': ['sum']})
act_center.columns = act_center.columns.droplevel(1)
act_center = act_center.rename(columns = {'t_airstrike_pred':'air_strike_center',
                              't_armor_pred':'armor_center',
                              't_raid_pred':'special_center',
                              't_cyber_pred':'cyber_center',
                              't_artillery_pred':'artillery_center'})

act_west = west.groupby(['date']).agg({'t_airstrike_pred': ['sum'],
                                           't_armor_pred': ['sum'],
                                           't_raid_pred': ['sum'],
                                           't_cyber_pred': ['sum'],
                                           't_artillery_pred': ['sum']})
act_west.columns = act_west.columns.droplevel(1)
act_west = act_west.rename(columns = {'t_airstrike_pred':'air_strike_west',
                              't_armor_pred':'armor_west',
                              't_raid_pred':'special_west',
                              't_cyber_pred':'cyber_west',
                              't_artillery_pred':'artillery_west'})

act_south = south.groupby(['date']).agg({'t_airstrike_pred': ['sum'],
                                           't_armor_pred': ['sum'],
                                           't_raid_pred': ['sum'],
                                           't_cyber_pred': ['sum'],
                                           't_artillery_pred': ['sum']})
act_south.columns = act_south.columns.droplevel(1)
act_south = act_south.rename(columns = {'t_airstrike_pred':'air_strike_south',
                              't_armor_pred':'armor_south',
                              't_raid_pred':'special_south',
                              't_cyber_pred':'cyber_south',
                              't_artillery_pred':'artillery_south'})

test = pd.merge(pd.merge(pd.merge(act_east[['air_strike_east', 'armor_east', 'special_east' ,'cyber_east', 'artillery_east']],
                act_center[['air_strike_center', 'armor_center', 'special_center' ,'cyber_center', 'artillery_center']],
                on='date', how='outer'), act_west[['air_strike_west', 'armor_west', 'special_west' ,'cyber_west', 'artillery_west']],
                         on='date', how='outer'), 
                act_south[['air_strike_south', 'armor_south', 'special_south' ,'cyber_south', 'artillery_south']], on='date', how='outer')
test = test.fillna(0)
act = test.copy()
act['Date'] = pd.to_datetime(act.index.values, format='%Y%m%d')
act = act.sort_values(by='Date')
act.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\act.csv", index=False)

test = test.cumsum(axis=0)
test['air_strike_east_prop'] = test['air_strike_east']/(test['air_strike_east']+test['air_strike_center']+test['air_strike_west']+test['air_strike_south'])
test['air_strike_center_prop'] = test['air_strike_center']/(test['air_strike_east']+test['air_strike_center']+test['air_strike_west']+test['air_strike_south'])
test['air_strike_west_prop'] = test['air_strike_west']/(test['air_strike_east']+test['air_strike_center']+test['air_strike_west']+test['air_strike_south'])
test['air_strike_south_prop'] = test['air_strike_south']/(test['air_strike_east']+test['air_strike_center']+test['air_strike_west']+test['air_strike_south'])

test['armor_east_prop'] = test['armor_east']/(test['armor_east']+test['armor_center']+test['armor_west']+test['armor_south'])
test['armor_center_prop'] = test['armor_center']/(test['armor_east']+test['armor_center']+test['armor_west']+test['armor_south'])
test['armor_west_prop'] = test['armor_west']/(test['armor_east']+test['armor_center']+test['armor_west']+test['armor_south'])
test['armor_south_prop'] = test['armor_south']/(test['armor_east']+test['armor_center']+test['armor_west']+test['armor_south'])

test['special_east_prop'] = test['special_east']/(test['special_east']+test['special_center']+test['special_west']+test['special_south'])
test['special_center_prop'] = test['special_center']/(test['special_east']+test['special_center']+test['special_west']+test['special_south'])
test['special_west_prop'] = test['special_west']/(test['special_east']+test['special_center']+test['special_west']+test['special_south'])
test['special_south_prop'] = test['special_south']/(test['special_east']+test['special_center']+test['special_west']+test['special_south'])

test['cyber_east_prop'] = test['cyber_east']/(test['cyber_east']+test['cyber_center']+test['cyber_west']+test['cyber_south'])
test['cyber_center_prop'] = test['cyber_center']/(test['cyber_east']+test['cyber_center']+test['cyber_west']+test['cyber_south'])
test['cyber_west_prop'] = test['cyber_west']/(test['cyber_east']+test['cyber_center']+test['cyber_west']+test['cyber_south'])
test['cyber_south_prop'] = test['cyber_south']/(test['cyber_east']+test['cyber_center']+test['cyber_west']+test['cyber_south'])

test['artillery_east_prop'] = test['artillery_east']/(test['artillery_east']+test['artillery_center']+test['artillery_west']+test['artillery_south'])
test['artillery_center_prop'] = test['artillery_center']/(test['artillery_east']+test['artillery_center']+test['artillery_west']+test['artillery_south'])
test['artillery_west_prop'] = test['artillery_west']/(test['artillery_east']+test['artillery_center']+test['artillery_west']+test['artillery_south'])
test['artillery_south_prop'] = test['artillery_south']/(test['artillery_east']+test['artillery_center']+test['artillery_west']+test['artillery_south'])
test = test.fillna(0)


op_prop = test
op_prop['air_strike_east_prop'].mask((op_prop['air_strike_east_prop'] == 0) &
                                     (op_prop['air_strike_center_prop'] == 0) &
                                     (op_prop['air_strike_west_prop'] == 0) & 
                                     (op_prop['air_strike_south_prop'] == 0), 'quarter', inplace=True)
op_prop['air_strike_center_prop'].mask((op_prop['air_strike_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['air_strike_west_prop'].mask((op_prop['air_strike_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['air_strike_south_prop'].mask((op_prop['air_strike_east_prop'] == 'quarter'), 'quarter', inplace=True)

op_prop['armor_east_prop'].mask((op_prop['armor_east_prop'] == 0) &
                                     (op_prop['armor_center_prop'] == 0) &
                                     (op_prop['armor_west_prop'] == 0) &
                                     (op_prop['armor_south_prop'] == 0), 'quarter', inplace=True)
op_prop['armor_center_prop'].mask((op_prop['armor_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['armor_west_prop'].mask((op_prop['armor_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['armor_south_prop'].mask((op_prop['armor_east_prop'] == 'quarter'), 'quarter', inplace=True)


op_prop['artillery_east_prop'].mask((op_prop['artillery_east_prop'] == 0) &
                                     (op_prop['artillery_center_prop'] == 0) &
                                     (op_prop['artillery_west_prop'] == 0) &
                                     (op_prop['artillery_south_prop'] == 0), 'quarter', inplace=True)
op_prop['artillery_center_prop'].mask((op_prop['artillery_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['artillery_west_prop'].mask((op_prop['artillery_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['artillery_south_prop'].mask((op_prop['artillery_east_prop'] == 'quarter'), 'quarter', inplace=True)


op_prop['special_east_prop'].mask((op_prop['special_east_prop'] == 0) &
                                     (op_prop['special_center_prop'] == 0) &
                                     (op_prop['special_west_prop'] == 0) &
                                     (op_prop['special_south_prop'] == 0), 'quarter', inplace=True)
op_prop['special_center_prop'].mask((op_prop['special_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['special_west_prop'].mask((op_prop['special_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['special_south_prop'].mask((op_prop['special_east_prop'] == 'quarter'), 'quarter', inplace=True)


op_prop['cyber_east_prop'].mask((op_prop['cyber_east_prop'] == 0) &
                                     (op_prop['cyber_center_prop'] == 0) &
                                     (op_prop['cyber_west_prop'] == 0) &
                                     (op_prop['cyber_south_prop'] == 0), 'quarter', inplace=True)
op_prop['cyber_center_prop'].mask((op_prop['cyber_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['cyber_west_prop'].mask((op_prop['cyber_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['cyber_south_prop'].mask((op_prop['cyber_east_prop'] == 'quarter'), 'quarter', inplace=True)


op_prop = op_prop.replace('quarter', 0.25)
op_prop['Dates'] = pd.to_datetime(op_prop.index.values, format='%Y%m%d')
op_prop['Dates']
op_prop = op_prop.sort_values(by='Dates')
Event_exp = op_prop
op_prop = op_prop.iloc[:,20:]

total_attack = test.iloc[-1][:20,].sum()
ip_air = (test['air_strike_east'].iloc[-1] + test['air_strike_center'].iloc[-1] +
          test['air_strike_west'].iloc[-1] + test['air_strike_south'].iloc[-1]) / total_attack
ip_at = (test['artillery_east'].iloc[-1] + test['artillery_center'].iloc[-1] + 
         test['artillery_west'].iloc[-1] + test['artillery_south'].iloc[-1]) / total_attack
ip_ar = (test['armor_east'].iloc[-1] + test['armor_center'].iloc[-1] +
         test['armor_west'].iloc[-1] + test['armor_south'].iloc[-1]) / total_attack
ip_sp = (test['special_east'].iloc[-1] + test['special_center'].iloc[-1] +
         test['special_west'].iloc[-1] + test['special_south'].iloc[-1]) / total_attack
ip_cb = (test['cyber_east'].iloc[-1] + test['cyber_center'].iloc[-1] + 
         test['cyber_west'].iloc[-1] + test['cyber_south'].iloc[-1]) / total_attack

op_prop['east_act'] = (ip_air*op_prop['air_strike_east_prop'] + ip_at*op_prop['artillery_east_prop'] +
                       ip_ar*op_prop['armor_east_prop'] + ip_sp*op_prop['special_east_prop'] +
                       ip_cb*op_prop['cyber_east_prop'])

op_prop['center_act'] = (ip_air*op_prop['air_strike_center_prop'] + ip_at*op_prop['artillery_center_prop'] +
                       ip_ar*op_prop['armor_center_prop'] + ip_sp*op_prop['special_center_prop'] +
                       ip_cb*op_prop['cyber_center_prop'])

op_prop['west_act'] = (ip_air*op_prop['air_strike_west_prop'] + ip_at*op_prop['artillery_west_prop'] +
                       ip_ar*op_prop['armor_west_prop'] + ip_sp*op_prop['special_west_prop'] +
                       ip_cb*op_prop['cyber_west_prop'])

op_prop['south_act'] = (ip_air*op_prop['air_strike_south_prop'] + ip_at*op_prop['artillery_south_prop'] +
                       ip_ar*op_prop['armor_south_prop'] + ip_sp*op_prop['special_south_prop'] +
                       ip_cb*op_prop['cyber_south_prop'])
op_prop.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\prop.csv", index=False)

plt.plot(op_prop['Dates'], op_prop['east_act'], label = 'East')
plt.plot(op_prop['Dates'], op_prop['center_act'], label = 'Center')
plt.plot(op_prop['Dates'], op_prop['west_act'], label = 'West')
plt.plot(op_prop['Dates'], op_prop['south_act'], label = 'South')
plt.legend(title = "Region", loc = 'upper right', bbox_to_anchor=(1.25, 1))
plt.xticks(rotation=45)
plt.xlabel('Date', fontsize=15)
plt.ylabel(r'$\pi$', fontsize=15)
plt.show()


##################### Divide dataset based on clustering result#####################
Event = event[['date', 'longitude', 'latitude', 'a_rus_pred','t_airstrike_pred', 't_armor_pred',
      't_raid_pred', 't_cyber_pred', 't_artillery_pred', 't_milcas_pred']]

a_list = []
for i in range(Event.shape[0]):
    if ((Event.iloc[i][1] <= 35.5) & (Event.iloc[i][2] >= 49)):
        a_list.append('center')
    else:
        a_list.append('east')

Event['region'] = a_list

Event['region'].mask(Event['longitude'] <= 27.5, 'west', inplace=True)
Event['region'].mask((Event['region'] == 'east') & (Event['longitude'] < 33.5) &
                     (Event['latitude'] > 48), 'center', inplace=True)
Event['region'].mask((Event['region'] == 'east') & (Event['longitude'] <= 36.8) &
                     (Event['latitude'] <= 48.8), 'south', inplace=True)


death = Event[['date', 'longitude', 'latitude', 't_milcas_pred', 'region']]
Event = Event.round({"longitude":3, "latitude":3, "a_rus_pred":0, "t_airstrike_pred":0, "t_armor_pred":0,
                     "t_raid_pred":0, "t_cyber_pred":0, 't_artillery_pred':0, 't_milcas_pred':0})

Event = Event.loc[Event['a_rus_pred'] == 1]
Event = Event.loc[((Event['t_airstrike_pred'] == 1) | (Event['t_armor_pred'] == 1) | 
                  (Event['t_raid_pred'] == 1) | (Event['t_cyber_pred'] == 1) |
                  (Event['t_artillery_pred'] == 1))]

Event.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\event.csv", index=False)
Event['region'].value_counts().plot(kind='bar')


Event_before = Event.loc[Event['date'] <= 20220325]
#Event_before = Event.loc[Event['date'] <= 20220615]
Event_before.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\event1.csv", index=False)
Event_before['region'].value_counts().plot(kind='bar')
act = []
for i in range(Event_before.shape[0]):
    if Event_before.iloc[i][4] == 1:
        act.append('air strike')
    elif Event_before.iloc[i][5] == 1:
        act.append('armor')
    elif Event_before.iloc[i][6] == 1:
        act.append('special')
    elif Event_before.iloc[i][7] == 1:
        act.append('cyber')
    elif Event_before.iloc[i][8] == 1:
        act.append('artillery')
Event_before['type'] = act

import matplotlib.pyplot as plt
import seaborn as sns
plots = sns.countplot(data=Event_before, x='type', hue='region')
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)

east = Event_before.loc[Event_before['region'] == 'east']
center = Event_before.loc[Event_before['region'] == 'center']
west = Event_before.loc[Event_before['region'] == 'west']
south = Event_before.loc[Event_before['region'] == 'south']

act_east = east.groupby(['date']).agg({'t_airstrike_pred': ['sum'],
                                           't_armor_pred': ['sum'],
                                           't_raid_pred': ['sum'],
                                           't_cyber_pred': ['sum'],
                                           't_artillery_pred': ['sum']})
act_east.columns = act_east.columns.droplevel(1)
act_east = act_east.rename(columns = {'t_airstrike_pred':'air_strike_east',
                              't_armor_pred':'armor_east',
                              't_raid_pred':'special_east',
                              't_cyber_pred':'cyber_east',
                              't_artillery_pred':'artillery_east'})

act_center = center.groupby(['date']).agg({'t_airstrike_pred': ['sum'],
                                           't_armor_pred': ['sum'],
                                           't_raid_pred': ['sum'],
                                           't_cyber_pred': ['sum'],
                                           't_artillery_pred': ['sum']})
act_center.columns = act_center.columns.droplevel(1)
act_center = act_center.rename(columns = {'t_airstrike_pred':'air_strike_center',
                              't_armor_pred':'armor_center',
                              't_raid_pred':'special_center',
                              't_cyber_pred':'cyber_center',
                              't_artillery_pred':'artillery_center'})

act_west = west.groupby(['date']).agg({'t_airstrike_pred': ['sum'],
                                           't_armor_pred': ['sum'],
                                           't_raid_pred': ['sum'],
                                           't_cyber_pred': ['sum'],
                                           't_artillery_pred': ['sum']})
act_west.columns = act_west.columns.droplevel(1)
act_west = act_west.rename(columns = {'t_airstrike_pred':'air_strike_west',
                              't_armor_pred':'armor_west',
                              't_raid_pred':'special_west',
                              't_cyber_pred':'cyber_west',
                              't_artillery_pred':'artillery_west'})

act_south = south.groupby(['date']).agg({'t_airstrike_pred': ['sum'],
                                           't_armor_pred': ['sum'],
                                           't_raid_pred': ['sum'],
                                           't_cyber_pred': ['sum'],
                                           't_artillery_pred': ['sum']})
act_south.columns = act_south.columns.droplevel(1)
act_south = act_south.rename(columns = {'t_airstrike_pred':'air_strike_south',
                              't_armor_pred':'armor_south',
                              't_raid_pred':'special_south',
                              't_cyber_pred':'cyber_south',
                              't_artillery_pred':'artillery_south'})

test = pd.merge(pd.merge(pd.merge(act_east[['air_strike_east', 'armor_east', 'special_east' ,'cyber_east', 'artillery_east']],
                act_center[['air_strike_center', 'armor_center', 'special_center' ,'cyber_center', 'artillery_center']],
                on='date', how='outer'), act_west[['air_strike_west', 'armor_west', 'special_west' ,'cyber_west', 'artillery_west']],
                         on='date', how='outer'), 
                act_south[['air_strike_south', 'armor_south', 'special_south' ,'cyber_south', 'artillery_south']], on='date', how='outer')
test = test.fillna(0)
act = test.copy()
act['Date'] = pd.to_datetime(act.index.values, format='%Y%m%d')
act = act.sort_values(by='Date')
act.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\act.csv", index=False)

test = test.cumsum(axis=0)
test['air_strike_east_prop'] = test['air_strike_east']/(test['air_strike_east']+test['air_strike_center']+test['air_strike_west']+test['air_strike_south'])
test['air_strike_center_prop'] = test['air_strike_center']/(test['air_strike_east']+test['air_strike_center']+test['air_strike_west']+test['air_strike_south'])
test['air_strike_west_prop'] = test['air_strike_west']/(test['air_strike_east']+test['air_strike_center']+test['air_strike_west']+test['air_strike_south'])
test['air_strike_south_prop'] = test['air_strike_south']/(test['air_strike_east']+test['air_strike_center']+test['air_strike_west']+test['air_strike_south'])

test['armor_east_prop'] = test['armor_east']/(test['armor_east']+test['armor_center']+test['armor_west']+test['armor_south'])
test['armor_center_prop'] = test['armor_center']/(test['armor_east']+test['armor_center']+test['armor_west']+test['armor_south'])
test['armor_west_prop'] = test['armor_west']/(test['armor_east']+test['armor_center']+test['armor_west']+test['armor_south'])
test['armor_south_prop'] = test['armor_south']/(test['armor_east']+test['armor_center']+test['armor_west']+test['armor_south'])

test['special_east_prop'] = test['special_east']/(test['special_east']+test['special_center']+test['special_west']+test['special_south'])
test['special_center_prop'] = test['special_center']/(test['special_east']+test['special_center']+test['special_west']+test['special_south'])
test['special_west_prop'] = test['special_west']/(test['special_east']+test['special_center']+test['special_west']+test['special_south'])
test['special_south_prop'] = test['special_south']/(test['special_east']+test['special_center']+test['special_west']+test['special_south'])

test['cyber_east_prop'] = test['cyber_east']/(test['cyber_east']+test['cyber_center']+test['cyber_west']+test['cyber_south'])
test['cyber_center_prop'] = test['cyber_center']/(test['cyber_east']+test['cyber_center']+test['cyber_west']+test['cyber_south'])
test['cyber_west_prop'] = test['cyber_west']/(test['cyber_east']+test['cyber_center']+test['cyber_west']+test['cyber_south'])
test['cyber_south_prop'] = test['cyber_south']/(test['cyber_east']+test['cyber_center']+test['cyber_west']+test['cyber_south'])

test['artillery_east_prop'] = test['artillery_east']/(test['artillery_east']+test['artillery_center']+test['artillery_west']+test['artillery_south'])
test['artillery_center_prop'] = test['artillery_center']/(test['artillery_east']+test['artillery_center']+test['artillery_west']+test['artillery_south'])
test['artillery_west_prop'] = test['artillery_west']/(test['artillery_east']+test['artillery_center']+test['artillery_west']+test['artillery_south'])
test['artillery_south_prop'] = test['artillery_south']/(test['artillery_east']+test['artillery_center']+test['artillery_west']+test['artillery_south'])
test = test.fillna(0)


op_prop = test
op_prop['air_strike_east_prop'].mask((op_prop['air_strike_east_prop'] == 0) &
                                     (op_prop['air_strike_center_prop'] == 0) &
                                     (op_prop['air_strike_west_prop'] == 0) & 
                                     (op_prop['air_strike_south_prop'] == 0), 'quarter', inplace=True)
op_prop['air_strike_center_prop'].mask((op_prop['air_strike_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['air_strike_west_prop'].mask((op_prop['air_strike_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['air_strike_south_prop'].mask((op_prop['air_strike_east_prop'] == 'quarter'), 'quarter', inplace=True)

op_prop['armor_east_prop'].mask((op_prop['armor_east_prop'] == 0) &
                                     (op_prop['armor_center_prop'] == 0) &
                                     (op_prop['armor_west_prop'] == 0) &
                                     (op_prop['armor_south_prop'] == 0), 'quarter', inplace=True)
op_prop['armor_center_prop'].mask((op_prop['armor_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['armor_west_prop'].mask((op_prop['armor_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['armor_south_prop'].mask((op_prop['armor_east_prop'] == 'quarter'), 'quarter', inplace=True)


op_prop['artillery_east_prop'].mask((op_prop['artillery_east_prop'] == 0) &
                                     (op_prop['artillery_center_prop'] == 0) &
                                     (op_prop['artillery_west_prop'] == 0) &
                                     (op_prop['artillery_south_prop'] == 0), 'quarter', inplace=True)
op_prop['artillery_center_prop'].mask((op_prop['artillery_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['artillery_west_prop'].mask((op_prop['artillery_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['artillery_south_prop'].mask((op_prop['artillery_east_prop'] == 'quarter'), 'quarter', inplace=True)


op_prop['special_east_prop'].mask((op_prop['special_east_prop'] == 0) &
                                     (op_prop['special_center_prop'] == 0) &
                                     (op_prop['special_west_prop'] == 0) &
                                     (op_prop['special_south_prop'] == 0), 'quarter', inplace=True)
op_prop['special_center_prop'].mask((op_prop['special_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['special_west_prop'].mask((op_prop['special_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['special_south_prop'].mask((op_prop['special_east_prop'] == 'quarter'), 'quarter', inplace=True)


op_prop['cyber_east_prop'].mask((op_prop['cyber_east_prop'] == 0) &
                                     (op_prop['cyber_center_prop'] == 0) &
                                     (op_prop['cyber_west_prop'] == 0) &
                                     (op_prop['cyber_south_prop'] == 0), 'quarter', inplace=True)
op_prop['cyber_center_prop'].mask((op_prop['cyber_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['cyber_west_prop'].mask((op_prop['cyber_east_prop'] == 'quarter'), 'quarter', inplace=True)
op_prop['cyber_south_prop'].mask((op_prop['cyber_east_prop'] == 'quarter'), 'quarter', inplace=True)


op_prop = op_prop.replace('quarter', 0.25)
op_prop['Dates'] = pd.to_datetime(op_prop.index.values, format='%Y%m%d')
op_prop['Dates']
op_prop = op_prop.sort_values(by='Dates')
Event_exp = op_prop
op_prop = op_prop.iloc[:,20:]

total_attack = test.iloc[-1][:20,].sum()
ip_air = (test['air_strike_east'].iloc[-1] + test['air_strike_center'].iloc[-1] +
          test['air_strike_west'].iloc[-1] + test['air_strike_south'].iloc[-1]) / total_attack
ip_at = (test['artillery_east'].iloc[-1] + test['artillery_center'].iloc[-1] + 
         test['artillery_west'].iloc[-1] + test['artillery_south'].iloc[-1]) / total_attack
ip_ar = (test['armor_east'].iloc[-1] + test['armor_center'].iloc[-1] +
         test['armor_west'].iloc[-1] + test['armor_south'].iloc[-1]) / total_attack
ip_sp = (test['special_east'].iloc[-1] + test['special_center'].iloc[-1] +
         test['special_west'].iloc[-1] + test['special_south'].iloc[-1]) / total_attack
ip_cb = (test['cyber_east'].iloc[-1] + test['cyber_center'].iloc[-1] + 
         test['cyber_west'].iloc[-1] + test['cyber_south'].iloc[-1]) / total_attack

op_prop['east_act'] = (ip_air*op_prop['air_strike_east_prop'] + ip_at*op_prop['artillery_east_prop'] +
                       ip_ar*op_prop['armor_east_prop'] + ip_sp*op_prop['special_east_prop'] +
                       ip_cb*op_prop['cyber_east_prop'])

op_prop['center_act'] = (ip_air*op_prop['air_strike_center_prop'] + ip_at*op_prop['artillery_center_prop'] +
                       ip_ar*op_prop['armor_center_prop'] + ip_sp*op_prop['special_center_prop'] +
                       ip_cb*op_prop['cyber_center_prop'])

op_prop['west_act'] = (ip_air*op_prop['air_strike_west_prop'] + ip_at*op_prop['artillery_west_prop'] +
                       ip_ar*op_prop['armor_west_prop'] + ip_sp*op_prop['special_west_prop'] +
                       ip_cb*op_prop['cyber_west_prop'])

op_prop['south_act'] = (ip_air*op_prop['air_strike_south_prop'] + ip_at*op_prop['artillery_south_prop'] +
                       ip_ar*op_prop['armor_south_prop'] + ip_sp*op_prop['special_south_prop'] +
                       ip_cb*op_prop['cyber_south_prop'])
op_prop.to_csv(r"C:\Users\gooyh\Documents\R\Ukraine\prop.csv", index=False)

plt.plot(op_prop['Dates'], op_prop['east_act'], label = 'East')
plt.plot(op_prop['Dates'], op_prop['center_act'], label = 'Center')
plt.plot(op_prop['Dates'], op_prop['west_act'], label = 'West')
plt.plot(op_prop['Dates'], op_prop['south_act'], label = 'South')
plt.legend(title = "Region", loc = 'upper right', bbox_to_anchor=(1.25, 1))
plt.xticks(rotation=45)
plt.xlabel('Date', fontsize=15)
plt.ylabel(r'$\pi$', fontsize=15)
plt.show()






