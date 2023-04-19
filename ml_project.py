# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:54:32 2023

@author: frank
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


df= pd.read_csv("C:\\Users\\frank\\Downloads\\asteroid_dataset.csv")
column_headers = list(df.columns.values)
target= df.iloc[:,-1]

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
print(tt)

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
plt.show()
    

    
    
    
# Check the correlation between size and speed
vd_corr= cor_df['Relative Velocity km per sec'].corr(cor_df['Est Dia in KM(max)'])
sns.scatterplot(x='Relative Velocity km per sec', y='Est Dia in KM(max)', data=cor_df)
plt.title('Correlation between Size and Speed')
plt.show()
print(f'The correlation between Relative Velocity km per sec and Est Dia in KM(max) is {vd_corr}')

me_corr= cor_df['Miss Dist. moon (au)'].corr(cor_df['Miss Dist. earth (au)'])
sns.scatterplot(x='Miss Dist. moon (au)', y='Miss Dist. earth (au)', data=cor_df)
plt.title('Correlation between Miss Dist. moon (au) and Miss Dist. earth (au)')
plt.show()
print(f'The correlation between Miss Dist. moon (au) and Miss Dist. earth (au) {me_corr}')

mm_corr= cor_df['Miss Dist. moon (au)'].corr(cor_df['Miss Dist.(Astronomical)'])
sns.scatterplot(x='Miss Dist. moon (au)', y='Miss Dist.(Astronomical)', data=cor_df)
plt.title('Correlation between Miss Dist. moon (au) and Miss Dist.(Astronomical)')
plt.show()
print(f'The correlation between Miss Dist. moon (au) km per sec and Miss Dist.(Astronomical) is {mm_corr}')
#thus we will us only Miss Dist.(Astronomical)

md_corr= cor_df['Est Dia in KM(max)'].corr(cor_df['Absolute Magnitude'])
sns.scatterplot(x='Est Dia in KM(max)', y='Absolute Magnitude', data=cor_df)
plt.title('Correlation between Est Dia in KM(max) and Absolute Magnitude')
plt.show()
print(f'The correlation between Est Dia in KM(max)and Absolute Magnitude {md_corr}')
# seems like the less size the more brightness STRANO

eu_corr= cor_df['Eccentricity'].corr(cor_df['Orbit Uncertainity'])
sns.scatterplot(x='Eccentricity', y='Orbit Uncertainity', data=cor_df)
plt.title('Correlation between Eccentricity and Orbit Uncertainity')
plt.show()
print(f'The correlation between Eccentricity and Orbit Uncertainityis {eu_corr}')
# no correlation

vm_corr= cor_df['Relative Velocity km per sec'].corr(cor_df['Mean Motion'])
sns.scatterplot(x='Relative Velocity km per sec', y='Mean Motion', data=cor_df)
plt.title('Correlation between Size and Mean Motion')
plt.show()
print(f'The correlation between Relative Velocity km per sec and Mean Motionis {vm_corr}')
#no corr

iu_corr= cor_df['Inclination'].corr(cor_df['Orbit Uncertainity'])
sns.scatterplot(x='Inclination', y='Orbit Uncertainity', data=cor_df)
plt.title('Correlation between Inclination and Orbit Uncertainity')
plt.show()
print(f'The correlation between Inclination and Orbit Uncertainityis {iu_corr}')
# no correlation

am_corr= cor_df['Mean Anomaly'].corr(cor_df['Mean Motion'])
sns.scatterplot(x='Mean Anomaly', y='Mean Motion', data=cor_df)
plt.title('Correlation between Mean Anomaly and Mean Motion')
plt.show()
print(f'The correlation between Mean Anomaly and Mean Motionis {am_corr}')
#no corr



# Count the number of hazardous NEOs
num_hazardous = df['Hazardous'].sum()
print(f'Total number of NEOs: {len(df)}')
print(f'Number of hazardous NEOs: {num_hazardous}')
print(f"Percentage of hazardous NEOs: {num_hazardous/len(df)*100:.2f}%")




############# undersampling with clustering 



#check for corerlations


df_false = df2[df2['Hazardous'] == False]


df_false = df_false.drop (['Neo Reference ID','Name','Perihelion Time','Asc Node Longitude','Epoch Date Close Approach','Epoch Osculation','Close Approach Date','Orbit Determination Date','Orbiting Body','Orbit ID','Equinox','Hazardous'], axis =1)

df_false['Est Dia in KM'] = (df_false['Est Dia in KM(min)']+df_false['Est Dia in KM(max)'])/2
df_false.drop('Est Dia in KM(min)', axis=1, inplace=True)
df_false.drop('Est Dia in KM(max)',axis=1, inplace=True)


# Preprocess the data by scaling the features
scaler = StandardScaler()
X_false = scaler.fit_transform(df_false.values)

# Fit the DBSCAN clustering algorithm to the preprocessed data
dbscan = DBSCAN(eps=1, min_samples=3, algorithm='ball_tree')
dbscan.fit(X_false)

# Extract the labels of the clusters
labels = dbscan.labels_

# Count the number of samples in each cluster
counts = np.bincount(labels[labels != -1])

# Select a subset of samples from each cluster to include in the final under-sampled dataset
samples_per_cluster = 100
selected_samples = []
for i in range(len(counts)):
    if counts[i] > samples_per_cluster:
        cluster_samples = df_false[labels == i].sample(samples_per_cluster)
    else:
        cluster_samples = df_false[labels == i]
    selected_samples.append(cluster_samples)
    
# Concatenate the selected samples into the final under-sampled dataset
under_sampled_df = pd.concat(selected_samples)
under_sampled_df['Hazardous'] = False

df_true = df2[df2['Hazardous'] == True]

df_true = df_true.drop (['Neo Reference ID','Name','Perihelion Time','Asc Node Longitude','Epoch Date Close Approach','Epoch Osculation','Close Approach Date','Orbit Determination Date','Orbiting Body','Orbit ID','Equinox'], axis =1)

df_true['Est Dia in KM'] = (df_true['Est Dia in KM(min)']+df_true['Est Dia in KM(max)'])/2
df_true.drop('Est Dia in KM(min)', axis=1, inplace=True)
df_true.drop('Est Dia in KM(max)',axis=1, inplace=True)


under_sampled_df=pd.concat([under_sampled_df, df_true], axis=0)


#----- Logistic Regression -----
response1= under_sampled_df.iloc[:,-1]
df_log_reg = under_sampled_df.iloc[: , :-1]

x_train, x_test, y_train, y_test = train_test_split(df_log_reg, response1, test_size=0.20, random_state=0)
scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.fit_transform(x_test)

logisticRegr = LogisticRegression(random_state=2409,penalty= 'none')
logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)

cm = metrics.confusion_matrix(y_test, predictions)
# Build the plot
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, annot_kws={'size':15},
            cmap=plt.cm.Oranges)
plt.title('Confusion Matrix for Logistic regression')
plt.show()


precision_score(y_test, predictions)

accuracy_score(y_test, predictions)

recall_score(y_test, predictions)

f1_score(y_test, predictions)

print(logisticRegr.coef_, logisticRegr.intercept_)

odds_ratios = np.exp(logisticRegr.coef_)

# Print the odds ratios
print("Odds Ratios: ", odds_ratios)
#the most significant feature is the 6

y_pred_proba = logisticRegr.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()



#----- Random Forest -----



x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(cor_df, target, test_size=0.20, random_state=0)

rf = RandomForestClassifier( max_depth=5).fit(x_train_rf, y_train_rf)

rf_pred = rf.predict(x_test_rf)


rf_accuracy_train= rf.score(x_train_rf, y_train_rf)
rf_accuracy = rf.score(x_test_rf, y_test_rf)
print(rf_accuracy) 

rf_cm = metrics.confusion_matrix(y_test_rf, rf_pred)

# Build the plot
plt.figure(figsize=(4,4))
sns.heatmap(rf_cm, annot=True, annot_kws={'size':15},
            cmap=plt.cm.Greens)


plt.title('Confusion Matrix for Random Forest Model')
plt.show()


