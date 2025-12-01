# evaluator.py

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(model, X_test, y_test, return_proba=True):
    """返回 accuracy, f1, auc"""

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    if return_proba:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except:
            auc = None
    else:
        auc = None

    return acc, f1, auc
