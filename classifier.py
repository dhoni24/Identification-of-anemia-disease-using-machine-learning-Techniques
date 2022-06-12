import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
df = pd.read_excel('D:\my project\AnemiaDiseasePrediction_Using_MachineLearning\AnemiaDiseasePrediction_Using_MachineLearning\src\SCA.xlsx', names = ["AGE","HB","HCT","RDW","MCV","MCH","MCHC","RBCcountinmillions","RETIC","HBF","HBAo","HBA2",'Diagnosis'])
features = ["AGE","HB","HCT","RDW","MCV","MCH","MCHC","RBCcountinmillions","RETIC","HBF","HBAo","HBA2"]

X = df.loc[1:,features].values
Y = df.loc[1:,['Diagnosis']].values.ravel()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.2, random_state = 0)
clfr = svm.SVC(kernel = 'linear', C = 0.1, probability=True ).fit(X_train, Y_train)
clfr1 = LogisticRegression(random_state=37,C=0.13).fit(X_train, Y_train)
clfr2 = KNeighborsClassifier(algorithm='brute', n_jobs=-1, n_neighbors=13, weights='uniform').fit(X_train, Y_train)
clfr3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0).fit(X_train, Y_train)
clfr4 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0).fit(X_train, Y_train)
joblib.dump(clfr, 'Anemia_model.pkl')

acc = clfr4.score(X_test, Y_test)
print("Accuracy: ",acc*100," %.")
