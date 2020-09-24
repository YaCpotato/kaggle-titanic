import pandas as pd
import pandas_profiling as pdp

import data_processing

X_train,Y_train,X_test,Y_test = data_processing.processed_data()

profile = pdp.ProfileReport(X_train)
profile.to_file("train.html")