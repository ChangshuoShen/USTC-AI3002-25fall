from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def get_base_models():
    return {
        "lr": LogisticRegression(max_iter=200),
        "svm": SVC(probability=True),
        "dt": DecisionTreeClassifier(max_depth=3),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "nb": GaussianNB(),
        "rf": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }
