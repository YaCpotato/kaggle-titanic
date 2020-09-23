import numpy as np
import pandas as pd

def row_read():
    """
    生データを読み込んで返すメソッド
    """
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train,test

def processed_data():
    """
    データの前処理をして返すメソッド
    """
    train,test = row_read()
    # あとから再度trainとtestに戻すので、並べかえはしない
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