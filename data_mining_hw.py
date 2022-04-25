import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

cancer = pd.read_csv("data.csv")
print(cancer.head(5), "\n")
cancer.columns
cancer.info()

cancer.drop(["Unnamed: 32"], axis=1, inplace=True)
print(cancer.head(3), "\n")

y = cancer['diagnosis'].values # Target variable
X = cancer.drop('diagnosis', axis =1).values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

print("DECISION TREE")
#Decision Tree
tree = DecisionTreeClassifier(criterion='entropy',max_depth= 3,random_state = 0)
tree.fit(X_train, y_train)
plot_tree(tree, feature_names = cancer.columns ,fontsize = 8)
y_pred_test = tree.predict(X_test)
print("Accuracy of Decision Tree: ", accuracy_score(y_test, y_pred_test), "\n")



# GINI IMPURITY
tree_gin_d4 = DecisionTreeClassifier(criterion = 'gini',max_depth =  4, random_state =  0 )
tree_gin_d4.fit(X_train, y_train)

y_pred_train_gin_d4 = tree_gin_d4.predict(X_train)
y_pred_test_gin_d4  = tree_gin_d4.predict(X_test)


# ENTROPY IMPURITY
tree_ent_d4 = DecisionTreeClassifier(criterion = 'entropy', max_depth =4,random_state =  0 )
tree_ent_d4.fit(X_train, y_train)

y_pred_train_ent_d4 = tree_ent_d4.predict(X_train)
y_pred_test_ent_d4  = tree_ent_d4.predict(X_test)

print("depth= 4")
print("GINI Accuracy  : ", accuracy_score(y_test, y_pred_test_gin_d4))
print("ENTROPY Accuracy  : ", accuracy_score(y_test, y_pred_test_ent_d4), "\n")

print("RANDOM FOREST")
model = RandomForestClassifier()
model.fit(X_train, y_train)
model_predictions = model.predict(X_test)
model_predictions[0:10]
y_test[0:10]
print("Accuracy of Random Forest: {}%".format(model.score(X_test, y_test) * 100 ), "\n")


