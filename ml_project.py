# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:47:54 2023

@author: frank
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df= pd.read_csv("C:\\Users\\frank\\Downloads\\asteroid_dataset.csv")
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
sns.heatmap(cor)
plt.show()

##### SARà GIUSTO ? TRARRE CONCLUSIONI

''' Analyzing the distributions '''
# Derving distribution info
tt=cor_df.describe()

# Now histograms
cor_df.hist(figsize=(15,15),bins=15)

#now Boxplots

f2 = plt.figure(figsize=(15, 15),constrained_layout=True)
gs = f2.add_gridspec(5, 8)

f2_ax1 = f2.add_subplot(gs[1,5 ])
sns.boxplot(x=cor_df['Miss Dist.(Astronomical)'])
f2_ax1.set_title('Miss Dist.(Astronomical)')

f2_ax1 = f2.add_subplot(gs[1, 1])
sns.boxplot(x=cor_df['Absolute Magnitude'])
f2_ax1.set_title('Absolute Magnitude')

f2_ax1 = f2.add_subplot(gs[1, 2])
sns.boxplot(x=cor_df['Est Dia in KM(min)'])
f2_ax1.set_title('Est Dia in KM(min)')

f2_ax1 = f2.add_subplot(gs[1, 3])
sns.boxplot(x=cor_df['Est Dia in KM(max)'])
f2_ax1.set_title('Est Dia in KM(max)')

f2_ax1 = f2.add_subplot(gs[1, 4])
sns.boxplot(x=cor_df['Relative Velocity km per sec'])
f2_ax1.set_title('Relative Velocity km per sec')

f2_ax1 = f2.add_subplot(gs[1, 6])
sns.boxplot(x=cor_df['Orbit Uncertainity'])
f2_ax1.set_title('Orbit Uncertainity')

f2_ax1 = f2.add_subplot(gs[1, 7])
sns.boxplot(x=cor_df['Minimum Orbit Intersection'])
f2_ax1.set_title('Minimum Orbit Intersection')

f2_ax1 = f2.add_subplot(gs[2, 1])
sns.boxplot(x=cor_df['Jupiter Tisserand Invariant'])
f2_ax1.set_title('Jupiter Tisserand Invariant')

f2_ax1 = f2.add_subplot(gs[2, 2])
sns.boxplot(x=cor_df['Eccentricity'])
f2_ax1.set_title('Eccentricity')

f2_ax1 = f2.add_subplot(gs[2, 3])
sns.boxplot(x=cor_df['Semi Major Axis'])
f2_ax1.set_title('Semi Major Axis')

f2_ax1 = f2.add_subplot(gs[2, 4])
sns.boxplot(x=cor_df['Inclination'])
f2_ax1.set_title('Inclination')

f2_ax1 = f2.add_subplot(gs[2, 5])
sns.boxplot(x=cor_df['Orbital Period'])
f2_ax1.set_title('Orbital Period')

f2_ax1 = f2.add_subplot(gs[2, 6])
sns.boxplot(x=cor_df['Perihelion Distance'])
f2_ax1.set_title('Perihelion Distance')

f2_ax1 = f2.add_subplot(gs[2, 7])
sns.boxplot(x=cor_df['Perihelion Arg'])
f2_ax1.set_title('Perihelion Arg')

f2_ax1 = f2.add_subplot(gs[3, 1])
sns.boxplot(x=cor_df['Aphelion Dist'])
f2_ax1.set_title('Aphelion Dist')

f2_ax1 = f2.add_subplot(gs[3, 2])
sns.boxplot(x=cor_df['Mean Anomaly'])
f2_ax1.set_title('Mean Anomaly')

f2_ax1 = f2.add_subplot(gs[3, 3])
sns.boxplot(x=cor_df['Mean Motion'])
f2_ax1.set_title('Mean Motion')

f2_ax1 = f2.add_subplot(gs[3, 4])
sns.boxplot(x=cor_df['Miss Dist. moon (au)'])
f2_ax1.set_title('Miss Dist. moon (au)')

f2_ax1 = f2.add_subplot(gs[3, 5])
sns.boxplot(x=cor_df['Miss Dist. earth (au)'])
f2_ax1.set_title('Miss Dist. earth (au)')
















































