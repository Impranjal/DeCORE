import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold



import numpy as np

# Load the dataset in a dataframe object and include only four features as mentioned
df = pd.read_csv("data3.csv")
#df['Annotation (0: Bot, 1: Normal customers, 2: Promotional customers, 3: Genuine users)'].replace(1,0,inplace=True)
df['Annotation (0: Bot, 1: Normal customers, 2: Promotional customers, 3: Genuine users)'].replace(2,0,inplace=True)
df['Annotation (0: Bot, 1: Normal customers, 2: Promotional customers, 3: Genuine users)'].replace(3,1,inplace=True)



include = ['Longeivity','Length of screen name','Does the profile have a description','Length of the description','Does the profile have a URL','Followee count of the user','Follower count of the user','Followee-by-follower ratio','Total number of tweets','Annotation (0: Bot, 1: Normal customers, 2: Promotional customers, 3: Genuine users)'] # Only four features
df_ = df[include]
print(df_)

# Data Preprocessing
print(df_.dtypes.iteritems())
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)
     print(categoricals)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
dependent_variable = 'Annotation (0: Bot, 1: Normal customers, 2: Promotional customers, 3: Genuine users)'

x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]



lr = LogisticRegression()
lr.fit(x, y)
model_accuracy=lr.score(x,y)
print(model_accuracy)

# Save your model
from sklearn.externals import joblib
joblib.dump(lr, 'mod1.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('mod1.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")