import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
from functools import partial
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2



# Load the data
df = pd.read_csv("heart_disease_cleaned.csv")
df.drop("thal", axis=1, inplace=True)

Data = df.drop("diagnosis", axis=1)

Target = df["diagnosis"]

mutual_info_classif_fixed = partial(mutual_info_classif, random_state=42)
Mutual_features = SelectKBest(mutual_info_classif_fixed, k=5)
Mutual_features.fit(Data, Target)

col = Mutual_features.get_support(indices=True)

Data = Data.iloc[:, col]

Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size=0.2, random_state=42)

print("Results Using Mutual Information")
#Decision Tree
DT_Classifier = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=15, min_samples_leaf=5)
DT_Classifier.fit(Data_train, Target_train)
DT_prediction = DT_Classifier.predict(Data_test)       
DT_accuracy = accuracy_score(Target_test, DT_prediction)
print("Decision Tree accuracy: ", DT_accuracy*100)
#Random Forest
RF_Classifier = RandomForestClassifier(n_estimators=5000, random_state=42,criterion="entropy")
RF_Classifier.fit(Data_train, Target_train)
RF_prediction = RF_Classifier.predict(Data_test)
RF_accuracy = accuracy_score(Target_test, RF_prediction)
print("Random Forest accuracy: ", RF_accuracy*100)

#SVM
SVM_Classifier = SVC(kernel="rbf", C=1,gamma=0.5)
SVM_Classifier.fit(Data_train, Target_train)
SVM_Prediction = SVM_Classifier.predict(Data_test)
SVM_accuracy = accuracy_score(Target_test, SVM_Prediction)
print("SVM accuracy: ", SVM_accuracy*100)

# Chi squared feature selection
print("\n\n")

Data = df.drop("diagnosis", axis=1)
Target = df["diagnosis"]

chi2_features = SelectKBest(chi2, k=5)
chi2_features.fit(Data, Target)

col = chi2_features.get_support(indices=True)

Data = Data.iloc[:, col]

Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size=0.2, random_state=42)

print("Results Using Chi Squared")
# Decision Tree
DT_Classifier = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=15, min_samples_leaf=5)
DT_Classifier.fit(Data_train, Target_train)
DT_prediction = DT_Classifier.predict(Data_test)       
DT_accuracy = accuracy_score(Target_test, DT_prediction)
print("Decision Tree accuracy: ", DT_accuracy*100)
# Random Forest
RF_Classifier = RandomForestClassifier(n_estimators=500, random_state=42, criterion="entropy")
RF_Classifier.fit(Data_train, Target_train)
RF_prediction = RF_Classifier.predict(Data_test)
RF_accuracy = accuracy_score(Target_test, RF_prediction)
print("Random Forest accuracy: ", RF_accuracy*100)

# SVM
SVM_Classifier = SVC(kernel="rbf", C=1, gamma=0.5)
SVM_Classifier.fit(Data_train, Target_train)
SVM_Prediction = SVM_Classifier.predict(Data_test)
SVM_accuracy = accuracy_score(Target_test, SVM_Prediction)
print("SVM accuracy: ", SVM_accuracy*100)