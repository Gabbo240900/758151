# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:54:32 2023

@author: frank
"""

import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
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
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn import tree
from IPython import display
import graphviz
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import plot_tree

df= pd.read_csv("C:\\Users\\frank\\Downloads\\asteroid_dataset.csv")
column_headers = list(df.columns.values)
target= df.iloc[:,-1]

# Dropping the columns withe distances in miles
df1=df.drop(['Est Dia in Miles(min)', 'Est Dia in Miles(max)','Est Dia in M(min)','Est Dia in M(max)','Est Dia in Feet(min)','Est Dia in Feet(max)','Relative Velocity km per hr','Miles per hour','Miss Dist.(miles)'], axis=1)                                       

# Checking for NAs
df.isnull().sum().sum()

ob=list(df['Orbiting Body'])
b=0
for o in ob:
    if o =='Earth':
        b+=1
# Orbiting body is the same far all the observations

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
fig_h = plt.figure(figsize=(17, 17))
ax_h = fig_h.add_subplot(1, 1, 1)
sns.heatmap(cor,cmap='coolwarm', center=0, annot= True, annot_kws={'size':11})
ax_h.set_title('Correlation matrix')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
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

fig = plt.figure(figsize=(20, 20), constrained_layout=True)
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
print(f'The correlation between Miss Dist. moon (au) and Miss Dist. earth (au) is {me_corr}')

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
print(f'The correlation between Est Dia in KM(max)and Absolute Magnitude is {md_corr}')
# seems like the less size the more brightness STRANO

eu_corr= cor_df['Eccentricity'].corr(cor_df['Orbit Uncertainity'])
sns.scatterplot(x='Eccentricity', y='Orbit Uncertainity', data=cor_df)
plt.title('Correlation between Eccentricity and Orbit Uncertainity')
plt.show()
print(f'The correlation between Eccentricity and Orbit Uncertainity is {eu_corr}')
# no correlation

vm_corr= cor_df['Relative Velocity km per sec'].corr(cor_df['Mean Motion'])
sns.scatterplot(x='Relative Velocity km per sec', y='Mean Motion', data=cor_df)
plt.title('Correlation between Size and Mean Motion')
plt.show()
print(f'The correlation between Relative Velocity km per sec and Mean Motion is {vm_corr}')
#no corr

iu_corr= cor_df['Inclination'].corr(cor_df['Orbit Uncertainity'])
sns.scatterplot(x='Inclination', y='Orbit Uncertainity', data=cor_df)
plt.title('Correlation between Inclination and Orbit Uncertainity')
plt.show()
print(f'The correlation between Inclination and Orbit Uncertainity is {iu_corr}')
# no correlation

am_corr= cor_df['Mean Anomaly'].corr(cor_df['Mean Motion'])
sns.scatterplot(x='Mean Anomaly', y='Mean Motion', data=cor_df)
plt.title('Correlation between Mean Anomaly and Mean Motion')
plt.show()
print(f'The correlation between Mean Anomaly and Mean Motionis {am_corr}')
#no corr


ju_corr= cor_df['Jupiter Tisserand Invariant'].corr(cor_df['Orbital Period'])
sns.scatterplot(x='Jupiter Tisserand Invariant', y='Orbital Period', data=cor_df)
plt.title('Correlation between Jupiter Tisserand Invariant and Orbital Period')
plt.show()
print(f'The correlation between Jupiter Tisserand Invariant and Orbital Periodis {ju_corr}')


# Count the number of hazardous NEOs
num_hazardous = df['Hazardous'].sum()
num_hazardouss = df['Hazardous'].reset_index()
sns.countplot(data=num_hazardouss,x='Hazardous')
plt.show()
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


num_hazardous_under = under_sampled_df['Hazardous'].sum()
num_hazardouss_under = under_sampled_df['Hazardous'].reset_index()
sns.countplot(data=num_hazardouss_under,x='Hazardous')
plt.show()
print(f'Total number of NEOs: {len(df)}')
print(f'Number of hazardous NEOs: {num_hazardous_under}')
print(f"Percentage of hazardous NEOs: {num_hazardous_under/len(under_sampled_df)*100:.2f}%")



#----- Logistic Regression -----

response1= under_sampled_df.iloc[:,-1]
df_log_reg = under_sampled_df.iloc[: , :-1]

lr_df1=cor_df
lr_df1['Est Dia in KM'] = (cor_df['Est Dia in KM(min)']+cor_df['Est Dia in KM(max)'])/2
lr_df1.drop('Est Dia in KM(min)', axis=1, inplace=True)
lr_df1.drop('Est Dia in KM(max)',axis=1, inplace=True)


x_train, x_test, y_train, y_test = train_test_split(df_log_reg, response1, test_size=0.20, random_state=0)
scaler = StandardScaler()

# get the indices of the rows in lr_df1 that also appear in df_log_reg
overlap_indices = lr_df1.index.intersection(x_train.index)

# drop the overlapping rows from lr_df1
x_test_tot = lr_df1.drop(overlap_indices)
y_test_tot= target.drop(overlap_indices)


scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test_tot)
scaled_test_bal =scaler.transform(x_test)

logisticRegr = LogisticRegression(random_state=2409,penalty= 'none')
logisticRegr.fit(scaled_train, y_train)

predictions = logisticRegr.predict(scaled_test)
score = logisticRegr.score(scaled_test, y_test_tot)
print(f'Score for unbalanced dataset{score}')

predictions_bal=logisticRegr.predict(scaled_test_bal)
score_bal= logisticRegr.score(scaled_test_bal, y_test)
print(f'Score for balanced dataset{score_bal}')


cm = metrics.confusion_matrix(y_test_tot, predictions)
# Build the plot
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, annot_kws={'size':15},
            cmap=plt.cm.Oranges)
plt.title('Confusion Matrix for Logistic regression (unbalanced)')
plt.xlabel('Predictions', fontsize=10)
plt.ylabel('Actuals', fontsize=10)
plt.show()


cm_bal = metrics.confusion_matrix(y_test, predictions_bal)
# Build the plot
plt.figure(figsize=(4,4))
sns.heatmap(cm_bal, annot=True, annot_kws={'size':15},
            cmap=plt.cm.Oranges)
plt.title('Confusion Matrix for Logistic regression (balanced)')
plt.xlabel('Predictions', fontsize=10)
plt.ylabel('Actuals', fontsize=10)
plt.show()




precision_lr =precision_score(y_test_tot, predictions)
print('Logistic regression precision (unbalanced):', precision_lr)

specificity_lr= specificity_score(y_test_tot, predictions)
print('Logistic regression specificity (unbalanced):', specificity_lr)

accuracy_lr=accuracy_score(y_test_tot, predictions)
print('Logistic regression accuracy (unbalanced):', accuracy_lr)

recall_lr=recall_score(y_test_tot, predictions)
print('Logistic regression recall (unbalanced):', recall_lr)

fi_lr=f1_score(y_test_tot, predictions)
print('Logistic regression F-1 score (unbalanced):', fi_lr)




precision_lr_bal =precision_score(y_test, predictions_bal)
print('Logistic regression precision (balanced):', precision_lr_bal)

specificity_lr_bal= specificity_score(y_test, predictions_bal)
print('Logistic regression specificity (balanced):', specificity_lr_bal)

accuracy_lr_bal=accuracy_score(y_test, predictions_bal)
print('Logistic regression accuracy (balanced):', accuracy_lr_bal)

recall_lr_bal=recall_score(y_test, predictions_bal)
print('Logistic regression recall (balanced):', recall_lr_bal)

fi_lr_bal=f1_score(y_test, predictions_bal)
print('Logistic regression F-1 score (balanced):', fi_lr_bal)







# print(logisticRegr.coef_, logisticRegr.intercept_)

# odds_ratios = logisticRegr.coef_
# odds=pd.DataFrame(odds_ratios)
# odds_transposed=odds.T
# odds_transposed.iloc
# plt.bar(odds_transposed.index.values, odds_transposed)

# # Show the plot
# plt.show()

# # Print the odds ratios
# print("Odds Ratios: ", odds_ratios)
# #the most significant feature is the 6



y_pred_proba = logisticRegr.predict_proba(scaled_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test_tot,  y_pred_proba)
auc = metrics.roc_auc_score(y_test_tot, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.title('ROC curve for Logistic regression (unbalanced)')
plt.show()




y_pred_proba_bal = logisticRegr.predict_proba(scaled_test_bal)[::,1]
fpr_bal, tpr_bal, _= metrics.roc_curve(y_test,  y_pred_proba_bal)
auc_bal = metrics.roc_auc_score(y_test, y_pred_proba_bal)

#create ROC curve
plt.plot(fpr_bal,tpr_bal,label="AUC="+str(auc_bal))
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.title('ROC curve for Logistic regression (balanced)')
plt.show()


import statsmodels.api as sm

response1= under_sampled_df.iloc[:,-1]
df_log_reg = under_sampled_df.iloc[: , :-1]

x_train_tot, x_test_tot, y_train_tot, y_test_tot = train_test_split(cor_df, target, test_size=0.20, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(df_log_reg, response1, test_size=0.20, random_state=0)
scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)
model = sm.Logit(y_train, scaled_train).fit()
params = model.params
conf = model.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
# convert log odds to ORs
odds = pd.DataFrame(np.exp(conf))
# check if pvalues are significant
odds['pvalues'] = model.pvalues
odds['significant?'] = ['significant' if pval <= 0.05 else 'not significant' for pval in model.pvalues]
odds





#----- Random Forest -----


x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(lr_df1, target, test_size=0.20, random_state=0)

rf = RandomForestClassifier(n_estimators=30, max_depth=8,max_features=3,min_samples_leaf=1, min_samples_split=12).fit(x_train_rf, y_train_rf)

rf_pred = rf.predict(x_test_rf)



rf_cm = metrics.confusion_matrix(y_test_rf, rf_pred)

# Build the plot
plt.figure(figsize=(4,4))
sns.heatmap(rf_cm, annot=True, annot_kws={'size':15},
            cmap=plt.cm.Greens)


plt.title('Confusion Matrix for Random Forest Model')
plt.show()


rf_accuracy =accuracy_score(y_test_rf, rf_pred)
print('Random forest accuracy:', rf_accuracy)

precision_rf= precision_score(y_test_rf, rf_pred)
print('Random forest precision:', precision_rf)

recall_rf = recall_score(y_test_rf, rf_pred)
print('Random forest recall:', recall_rf)

f1_rf= f1_score(y_test_rf, rf_pred)
print('Random forest F-1 score:', f1_rf)


y_pred_proba_rf = rf.predict_proba(x_test_rf)[:, 1]
auc_rf = metrics.roc_auc_score(y_test_rf, y_pred_proba_rf)

# calculate ROC curve of the model
fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test_rf, y_pred_proba_rf)

# plot the ROC curve
plt.plot(fpr_rf, tpr_rf, label='ROC curve (AUC = {:.2f})'.format(auc_rf))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4 ,5 ,6, 7, 8, 10, 12],
    'n_estimators': [10, 20, 30, 40, 50]
}



# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(x_train_rf, y_train_rf)
print(grid_search.best_params_)


  
for t in rf.estimators_[:5]:
    fig, ax = plt.subplots(figsize=(10, 10),dpi=300)
    plot_tree(t, filled=True, ax=ax)
plt.show()
        


imp= permutation_importance(rf, x_test_rf, y_test_rf)
feature_names = [cor_df.columns.values]
forest_importances = pd.Series(imp.importances_mean, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=imp.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

# tree_importances = []
# for tree in rf.estimators_:
#     importances = tree.tree_.compute_feature_importances(normalize=False)
#     tree_importances.append(importances)

# # Compute the mean importance scores across all trees
# mean_importances = np.mean(tree_importances, axis=0)

# # Sort the features by importance score
# sorted_indices = mean_importances

# # Print the top 5 most important trees
# for i in range(5):
#     print(f"Tree {i+1}: Importance score = {mean_importances[sorted_indices[i]]:.2f}")

#----- Neural network -----

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(24, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
#[classifier.add(Dense(12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the fifth hidden layer
classifier.add(Dense(12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy','Recall'])

# Fitting the ANN to the Training set
classifier.fit(x_train_rf, y_train_rf, batch_size = 15, epochs = 300)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test_rf)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_rf, y_pred)

# Computing Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("Accuracy =", accuracy_score(y_test_rf,y_pred))
print("Precision = ", precision_score(y_test_rf,y_pred))
print("Recall = ", recall_score(y_test_rf,y_pred))
