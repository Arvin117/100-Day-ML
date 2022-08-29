from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,\
                           random_state=0, shuffle=False)

# clf = AdaBoostClassifier(n_estimators=100, random_state=0)    # 默认准确率0.983
cl = SVC()  # 初始化个自己的弱学习器, 准确率为0.504
clf = AdaBoostClassifier(base_estimator=cl, n_estimators=100, random_state=0, algorithm='SAMME')
clf.fit(X, y)

cl.fit(X, y)
print(cl.score(X, y))  # 单独用SVM准确率为0.958
# 这代表多个准确率高的分类器集成之后的效果可能会更差
clf.predict([[0, 0, 0, 0]])
print(clf.score(X, y))