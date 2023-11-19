import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']
df = pd.read_csv('adult.data', header=None, names = col_names)

#Distribution of income
print('\nIncome Distribution')
print(df.income.value_counts(normalize=True))

#Clean columns by stripping extra whitespace for columns of type "object"
for col in df.select_dtypes(include=['object']):
  df[col] = df[col].str.strip()

feature_cols = ['age',
       'capital-gain', 'capital-loss', 'hours-per-week', 'sex','race']
#Create feature dataframe X with feature columns and dummy variables for categorical features
X = pd.get_dummies(df[feature_cols], drop_first=True)

#Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greather than 50k
y = np.where(df.income == '<=50K', 0, 1)

#Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size=0.2 )

#Instantiate random forest classifier, fit and score with default parameters
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

print('\nDefault Depth Score: ')
print(rf.score(x_test, y_test))



#Tune the hyperparameter max_depth over a range from 1-25, save scores for test and train set
np.random.seed(0)
accuracy_train=[]
accuracy_test = []
depths = range(1, 26)
for i in depths:
  rf = RandomForestClassifier(max_depth=i)
  rf.fit(x_train, y_train)
  accuracy_test.append(rf.score(x_test, y_test))
  accuracy_train.append(rf.score(x_train, y_train))


    
#Find the best accuracy and at what depth that occurs
max_accuracy_test = max(accuracy_test)
max_accuracy_test_depth = accuracy_test.index(max_accuracy_test) 

print('\nBest Depth: ')
print(max_accuracy_test_depth)

#Plot the accuracy scores for the test and train set over the range of depth values  
plt.plot(depths, accuracy_test, depths, accuracy_train)
plt.legend(['test accuracy', 'train accuracy'])
plt.title('Accuracy vs Depth')
plt.xlabel('max depths')
plt.ylabel('accuracy')
plt.show()
plt.clf()

#Save the best random forest model and save the feature importances in a dataframe
best_rf = RandomForestClassifier(max_depth=max_accuracy_test_depth)
best_rf.fit(x_train, y_train)
feature_importances = best_rf.feature_importances_
feature_names = x_train.columns
feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importances_df.sort_values(by='Importance', ascending=False)
print('\nTop 5 Features: ')
print(feature_importance_df.head(5))

#Create two new features, based on education and native country
df['education_bin'] = pd.cut(df['education-num'], [0,9,13,16], labels=['HS or less', 'College to Bachelors', 'Masters or more'])

feature_cols = ['age',
        'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race','education_bin']
#Use these two new additional features and recreate X and test/train split
X = pd.get_dummies(df[feature_cols], drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size=0.2 )

accuracy_train=[]
accuracy_test = []
depths = range(1, 26)
for i in depths:
  rf = RandomForestClassifier(max_depth=i)
  rf.fit(x_train, y_train)
  accuracy_test.append(rf.score(x_test, y_test))
  accuracy_train.append(rf.score(x_train, y_train))

#Find the best max depth now with the additional two features
max_accuracy_test = max(accuracy_test)
max_accuracy_test_depth = accuracy_test.index(max_accuracy_test) 

print('\nBest Depth (new): ')
print(max_accuracy_test_depth)

#Save the best model and print the two features with the new feature set
plt.figure(2)
plt.plot(depths, accuracy_test, depths, accuracy_train)
plt.legend(['test accuracy', 'train accuracy'])
plt.title('Accuracy vs Depth (new)')
plt.xlabel('max depths')
plt.ylabel('accuracy')
plt.show()

best_rf_new = RandomForestClassifier(max_depth=max_accuracy_test_depth)
best_rf_new.fit(x_train, y_train)
feature_importances = best_rf_new.feature_importances_
feature_names = x_train.columns
feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importances_df.sort_values(by='Importance', ascending=False)
print('\nTop 5 Features: ')
print(feature_importance_df.head(5))

