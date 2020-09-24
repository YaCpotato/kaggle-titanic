import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

import data_processing

X_train,Y_train,X_test,Y_test = data_processing.processed_data()

clf = LogisticRegression(penalty = 'l2', solver = 'sag', random_state = 0, verbose = 1)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
#print(Y_test)
#print (confusion_matrix(Y_test, Y_pred))
#print (classification_report(Y_test, Y_pred))

sub = pd.read_csv('./gender_submission.csv')
sub['Survived'] = list(map(int, Y_pred))
sub.to_csv('submission.csv', index=False)
