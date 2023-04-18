import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
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
    
    
    
    
    
# Check the correlation between size and speed
sns.scatterplot(x='Relative Velocity km per hr', y='Est Dia in KM(max)', data=df)
plt.title('Correlation between Size and Speed')
plt.show()
###DOBBIAMO VEDERE ALTRE CORRELAZIONI INTERERSSANTI########################    


# Count the number of hazardous NEOs
num_hazardous = df['Hazardous'].sum()
print(f'Total number of NEOs: {len(df)}')
print(f'Number of hazardous NEOs: {num_hazardous}')
print(f"Percentage of hazardous NEOs: {num_hazardous/len(df)*100:.2f}%")




############# undersampling with clustering 



#check for corerlations

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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



































# # Get the data types of each column
# dtypes = df2.dtypes

# # Identify columns with obj type
# obj_cols = [col for col, dtype in dtypes.items() if dtype == 'object']

# # Remove columns with obj type
# df3 = df2.drop(obj_cols, axis=1)


# # Split the data into training and testing sets
# X = df3.drop('Hazardous', axis=1) # Features
# y = df3['Hazardous'] # Target variable
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the data to improve model performance
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)




# # Train a logistic regression model
# lr_model = LogisticRegression(random_state=42)
# lr_model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = lr_model.predict(X_test)

# # Evaluate the model using accuracy, precision, recall, and F1 score
# lr_precision = precision_score(y_test, y_pred)
# lr_recall = recall_score(y_test, y_pred)
# lr_f1 = f1_score(y_test, y_pred)
# lr_accuracy = accuracy_score(y_test, y_pred)

# print(f"Logistic Regression Model:\n Accuracy: {lr_accuracy:.2f}\n Precision: {lr_precision:.2f}\n Recall: {lr_recall:.2f}\n F1 Score: {lr_f1:.2f}\n")
    
    
    

# # Train a random forest classifier
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf_model.predict(X_test)

# # Evaluate the model using accuracy, precision, recall, and F1 score
# rf_precision = precision_score(y_test, y_pred)
# rf_recall = recall_score(y_test, y_pred)
# rf_f1 = f1_score(y_test, y_pred)
# rf_accuracy = accuracy_score(y_test, y_pred)

# print(f"Random Forest Classifier Model:\n Accuracy: {rf_accuracy:.2f}\n Precision: {rf_precision:.2f}\n Recall: {rf_recall:.2f}\n F1 Score: {rf_f1:.2f}\n")


# #Train the Gradient Boosting model
# gb_model = GradientBoostingClassifier(random_state=42)
# gb_model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = gb_model.predict(X_test)

# # Evaluate the model using accuracy, precision, recall, and F1 score
# gb_precision = precision_score(y_test, y_pred)
# gb_recall = recall_score(y_test, y_pred)
# gb_f1 = f1_score(y_test, y_pred)
# gb_accuracy = accuracy_score(y_test, y_pred)

# print(f"Gradient Boosting Classifier Model:\n Accuracy: {gb_accuracy:.2f}\n Precision: {gb_precision:.2f}\n Recall: {gb_recall:.2f}\n F1 Score: {gb_f1:.2f}\n")





    
    
