# models.py (Student Version)

from sklearn.svm import SVC

def get_linear_svm(C):
    return SVC(kernel="linear", C=C, probability=True)

def get_rbf_svm(C, gamma):
    return SVC(kernel="rbf", C=C, gamma=gamma, probability=True)

def get_poly_svm(C, degree, gamma=1.0, coef0=1.0):
    return SVC(kernel="poly", C=C, degree=degree, gamma=gamma, coef0=coef0, probability=True)
