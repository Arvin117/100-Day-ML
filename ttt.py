import numpy as np
import pylab as pl
from sklearn import svm

# 每次随机数据相同
np.random.seed(0)

x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20

clf = svm.SVC(kernel='linear')
clf.fit(x, y)

w = clf.coef_[0]
a = - w[0] / w[1]
xx = np.linspace(-5, 5)

# 所求最大间隔分界线
yy = a * xx - (clf.intercept_[0] / w[1])
# 最大间隔下面的线
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
# 最大间隔上面的线
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# 打印出参数

print("w: ", w)
print("a: ", a)
print("support_vectors_: ", clf.support_vectors_)
print("clf.coef_: ", clf.coef_)

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

# 把支持向量圈起来
pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')

# 画出各个点
pl.scatter(x[:, 0], x[:, 1], c=y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
