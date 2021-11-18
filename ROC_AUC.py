import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import tree, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score



def plot_2dscatter(X,y):
    plt.subplot()
    plt.title("One informative feature, one cluster per class", fontsize="small")
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
    plt.show()
    
X,y=datasets.make_gaussian_quantiles(n_features=2,cov=3,n_samples=500,n_classes=2)
plot_2dscatter(X,y)






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf = tree.DecisionTreeClassifier(min_samples_leaf=3)
clf = clf.fit(X_train, y_train)
y_score=clf.predict_proba(X_test)
y_predict=clf.predict(X_test)
plot_2dscatter(X_test,y_predict)


def bin_roc(y_score,y_test):
    
    fpr, tpr, threshold = roc_curve(y_test, y_score)
   
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc


bin_roc(y_score[:,1],y_test)








