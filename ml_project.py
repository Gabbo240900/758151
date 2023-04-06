# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:47:54 2023

@author: frank
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df= pd.read_csv("asteroid_dataset.csv")
column_headers = list(df.columns.values)

# Dropping the columns withe distances in miles
df1=df.drop(['Est Dia in Miles(min)', 'Est Dia in Miles(max)','Est Dia in M(min)','Est Dia in M(max)','Est Dia in Feet(min)','Est Dia in Feet(max)','Relative Velocity km per hr','Miles per hour','Miss Dist.(miles)'], axis=1)                                       

# Checking for NAs
df.isnull().sum().sum()

# Checking for duplicates 
a=list(df.duplicated(keep=False))
c=0
for i in a :
    if i==True:
        c=c+1
# Now checking if the same object appears multiple times in the df

b=list(df1.duplicated(subset= ['Neo Reference ID'],keep=False))       
d=0
for l in b :
    if l==True:
        d=d+1
# Some objects are repeating

''' Feature engeneering '''
# Not all variables are expressed in the same unit of measure so we will cast the ones that are reffered to a distance into AU

miss_dist_from_moon_au= df1['Miss Dist.(lunar)'] * 0.0025695686589742 
    
miss_dist_from_earth_au=(df['Miss Dist.(kilometers)']) * 6.684587122671e-9

df2=df1.drop(['Miss Dist.(lunar)', 'Miss Dist.(kilometers)'], axis=1)      
df2['Miss Dist. moon (au)']=miss_dist_from_moon_au
df2['Miss Dist. earth (au)']= miss_dist_from_earth_au

##### RIVEDERE APPROSSIMAZIONI

#check for corerlations 
cor_df = df2.drop (['Neo Reference ID','Name','Perihelion Time','Asc Node Longitude','Epoch Date Close Approach','Epoch Osculation','Close Approach Date','Orbit Determination Date','Orbiting Body','Orbit ID','Equinox','Hazardous'], axis =1)
column_headers2 = list(cor_df.columns.values)
cor = cor_df.corr()
sns.heatmap(cor,cmap='coolwarm', center=0)
plt.show()

##### SARà GIUSTO ? TRARRE CONCLUSIONI

''' Analyzing the distributions '''
# Derving distribution info
tt=cor_df.describe()

# Histogram
fig, ax = plt.subplots(figsize=(15, 15))
cor_df.hist(ax=ax, bins=15)
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Asteroid Features')

# Boxplots
columns = [
    'Miss Dist.(Astronomical)', 'Absolute Magnitude', 'Est Dia in KM(min)',
    'Est Dia in KM(max)', 'Relative Velocity km per sec', 'Orbit Uncertainity',
    'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant', 'Eccentricity',
    'Semi Major Axis', 'Inclination', 'Orbital Period', 'Perihelion Distance',
    'Perihelion Arg', 'Aphelion Dist', 'Mean Anomaly', 'Mean Motion',
    'Miss Dist. moon (au)', 'Miss Dist. earth (au)'
]

def plot_boxplot(column):
    fig_ax = fig.add_subplot(gs[i, j])
    sns.boxplot(x=cor_df[column], ax=fig_ax)
    fig_ax.set_title(column)

fig = plt.figure(figsize=(15, 15), constrained_layout=True)
gs = fig.add_gridspec(5, 8)

for idx, col in enumerate(columns):
    i, j = divmod(idx, 8)
    plot_boxplot(col)
