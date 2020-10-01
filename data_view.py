import pandas as pd
import pandas_profiling as pdp

import data_processing

X_train,Y_train,X_test,Y_test = data_processing.processed_data()
df = pd.read_csv('train.csv')

pdf = pdp.ProfileReport(df)
profile = pdp.ProfileReport(X_train)
profile.to_file('train_after_creansing.html')
pdf.to_file('train.html')