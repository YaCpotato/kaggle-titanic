import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('gender_submission.csv')

data = pd.concat([train, test], sort = False)
data['Sex'].replace(['male', 'female'],[0,1],inplace = True)
data['Fare'].fillna(np.mean(data['Fare']), inplace = True)
data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]

Y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
Y_test = test['Survived']
X_test = test.drop('Survived', axis=1)

clf = LogisticRegression(penalty = 'l2', solver = 'sag', random_state = 0, verbose = 1)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
#print(Y_test)
#print (confusion_matrix(Y_test, Y_pred))
#print (classification_report(Y_test, Y_pred))

sub = pd.read_csv('./gender_submission.csv')
sub['Survived'] = list(map(int, Y_pred))
sub.to_csv('submission.csv', index=False)
