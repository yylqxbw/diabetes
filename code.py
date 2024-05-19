import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler

Dataset = pd.read_csv('diabetes.csv')
Dataset

Dataset.describe()

Dataset.groupby('Outcome').size()

Dataset.hist(figsize=(16, 14))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.suptitle("分布直方图-田浩辰", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

corr = Dataset.corr()
corr

plt.subplots(figsize=(14,12))
sns.heatmap(corr, annot = True)
plt.suptitle("热点图-田浩辰", fontsize=20)

X = Dataset.iloc[:, 0:8]
Y = Dataset.iloc[:, 8]
select_top_5 = SelectKBest(score_func=chi2, k=5)
fit = select_top_5.fit(X, Y)
features = fit.transform(X)
features

X_features = pd.DataFrame(data = features, columns=['Pregnancies','Glucose','Insulin','BMI','Age'])
X_features

rescaledX = StandardScaler().fit_transform(X_features)
X = pd.DataFrame(data=rescaledX, columns=X_features.columns)
X

from sklearn.model_selection import train_test_split
Dataset.info()
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=2024, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 10)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    training_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))

plt.figure()
plt.plot(neighbors_settings, training_accuracy, label="training set")
plt.plot(neighbors_settings, test_accuracy, label="test set")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.title("比较KNN在不同参数下训练集和测试集的不同表现-陈至立")

print("training set: {:.2f}".format(knn.score(x_train, y_train)))
print("test set: {:.2f}".format(knn.score(x_test, y_test)))

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear').fit(x_train, y_train)
print("Training set score : {:.3f}".format(logreg.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(x_test, y_test)))

logreg100 = LogisticRegression(C=100, solver='liblinear').fit(x_train, y_train)
print("Training set score : {:.3f}".format(logreg100.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(x_test, y_test)))

logreg001 = LogisticRegression(C=0.001, solver='liblinear').fit(x_train, y_train)
print("Training set score : {:.3f}".format(logreg001.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(x_test, y_test)))

diabetes_features = [x for i, x in enumerate(X.columns) if i != 5]

plt.figure(figsize=(8, 6))
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, 'd', label="C=100")
plt.plot(logreg001.coef_.T, '*', label="C=0.001")
plt.xticks(range(X.shape[1]), diabetes_features, rotation=90)
plt.hlines(0, 0, X.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.title("不同正则化参数下所得的模型系数-程玉麟")

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)
print("set: {:.3f}".format(tree.score(x_train, y_train)))
print("test set: {:.3f}".format(tree.score(x_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(x_train, y_train)
print("training set: {:.3f}".format(tree.score(x_train, y_train)))
print("test set: {:.3f}".format(tree.score(x_test, y_test)))

def plot_feature_importances_diatebes(model):
    plt.figure(figsize=(8, 6))
    n_features = 5
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Features importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title("各参数重要程度-任惠飞")

plot_feature_importances_diatebes(tree)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf = RandomForestClassifier(max_depth=7, n_estimators=100, random_state=0)
rf.fit(x_train, y_train)

print("training set: {:.3f}".format(rf.score(x_train, y_train)))
print("test set: {:.3f}".format(rf.score(x_test, y_test)))

plot_feature_importances_diatebes(rf)
plt.title("各参数重要程度-赵翔宇")

x_train = x_train.astype(np.float64)
y_train = y_train.astype(np.float64)
x_test = x_test.astype(np.float64)
y_test = y_test.astype(np.float64)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
print("training set: {:.2f}".format(svc.score(x_train, y_train)))
print("test set: {:.2f}".format(svc.score(x_test, y_test)))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
svc = SVC()
svc.fit(x_train_scaled, y_train)
print("training set: {:.2f}".format(svc.score(x_train_scaled, y_train)))
print("test set: {:.2f}".format(svc.score(x_test_scaled, y_test)))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid

tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(5,)),     # @REPLACE
        Dense(5, activation='sigmoid', name = "L1"),
        Dense(3, activation='sigmoid', name = "L2"),
        Dense(1, activation='sigmoid',  name = "L3"), # @REPLACE
    ], name = "my_model"
)

model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.02)
    )

history = model.fit(
        x_train, y_train,
        epochs=100
    )

def Testaccuracy():
    prediction = model.predict(x_test)
    num = 0
    m = prediction.size
    ans = y_test.tolist()
    for i in range(m):
        yhat = 0
        if(prediction[i][0]>=0.5):
            yhat = 1
        else:
            yhat = 0
        if(yhat == ans[i]):
            num = num + 1
    return num/m
def Trainaccuracy():
    prediction = model.predict(x_train)
    num = 0
    m = prediction.size
    ans = y_train.tolist()
    for i in range(m):
        yhat = 0
        if(prediction[i][0]>=0.5):
            yhat = 1
        else:
            yhat = 0
        if(yhat == ans[i]):
            num = num + 1
    return num/m
print(f'测试集准确率{Testaccuracy()}')
print(f'训练集准确率{Trainaccuracy()}')

import matplotlib.pyplot as plt

models = ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC', 'Neural Network']

train_accuracy = [0.79, 0.764, 0.803, 0.914, 0.78, Trainaccuracy()]
test_accuracy = [0.73, 0.773, 0.747, 0.779, 0.77, Testaccuracy()]


plt.figure(figsize=(10, 6))

plt.plot(models, train_accuracy, marker='o', label='Training set Accuracy')

plt.plot(models, test_accuracy, marker='o', label='Test set Accuracy')

plt.title('模型精度比较-陈至立')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1)

plt.legend()

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

