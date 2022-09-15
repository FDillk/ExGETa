import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pandas

modelpath = "/modelsave/model2"
iono = pandas.read_csv('foo.csv', sep=",")
traintestcutoff = 200

Y = iono.iloc[:, :1].values
X = iono.iloc[:, 1:].values

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
X = StandardScaler().fit_transform(X)

X_train = X[:traintestcutoff, :]
X_test = X[traintestcutoff:, :]

y_train = Y[:traintestcutoff]
y_test = Y[traintestcutoff:]


clf = SVC(C=0.5, kernel="linear").fit(X_train,y_train)

# Save to file in the current working directory
pkl_filename = "pickle_iono.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)
print("Pickle saved")

joblib_file = "joblib_iono.pkl"
joblib.dump(clf, joblib_file)
print("Joblib saved")

print(clf.score(X_test, y_test))

