from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import dump, load


train_size = 20000
val_size = 10000
test_size = 10000
total_size = train_size+val_size+test_size

datafile = "../../wristdata1024_raw.csv.gz"
data = pd.read_csv(datafile, nrows=total_size)
print(data.shape)

x_cols = [s for s in data.columns if s.startswith("s1") or s.startswith("s2")]
y_col = "movement"
X_train = data.loc[:train_size-1, x_cols].values
y_train = data.loc[:train_size-1, y_col].values
X_val = data.loc[train_size:train_size+val_size-1, x_cols].values
y_val = data.loc[train_size:train_size+val_size-1, y_col].values
X_test = data.loc[train_size+val_size:total_size, x_cols].values
y_test = data.loc[train_size+val_size:total_size, y_col].values
del data
print(X_train.shape, X_val.shape, X_test.shape)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(f"train score: {train_score}")
print(f"test score: {test_score}")

dump(clf, 'wristdata1024_raw_svm_classifier.joblib')
