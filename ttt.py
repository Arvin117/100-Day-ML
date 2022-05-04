#! usr/bin/python <br> # -*- coding:utf8 -*-
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
def plot_decision_regions(X, y, classifier,test_idx = None, resolution=0.02):
    #setup marker generator and colormap
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[: len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:,0].min() -1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min() -1, X[:,1].max()+1
    # X[:,k] 冒号左边表示行范围，读取所有行，冒号右边表示列范围，读取第K列
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    #arange(start,end,step) 返回一个一维数组
    #meshgrid(x,y)产生一个以x为行，y为列的矩阵
    #xx1是一个(305*235)大小的矩阵 xx1.ravel()是将所有的行放在一个行里面的长度71675的一维数组
    #xx2同理
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #np.array([xx1.ravel(), xx2.ravel()]) 生成了一个 (2*71675)的矩阵
    # xx1.ravel() = (1,71675)
    #xx1.shape = (305,205) 将Z重新调整为(305,205)的格式
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    print(np.unique(y))
    # idx = 0,1 cl = -1 1
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker = markers[idx],label = cl)
    #highlight test samples   
    #增加的模块
    if test_idx:
        X_test, y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='',edgecolors='0',
                    alpha=1.0, linewidths=1,marker='o',
                    s=55, label='test set')