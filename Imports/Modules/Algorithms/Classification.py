from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def build_NB():
    """Builds a Naive Bayes Classifier

    Returns:
        GaussianNB() -- A Gaussian Naive Bayes Classifier with default parameters
    """
    return GaussianNB()

def build_knn():
    """Nearest Neighbors algorithm with parameters selected after grid searching on limited data

    Returns:
        KNN Classifier -- Classifier to be  fitted on data
    """
    gs = KNeighborsClassifier(n_neighbors = 13)
    return gs


def build_ensemble():
    """Builds an ensemble classifier of the random forest type at the moment.
    Returns:

     Ensemble classifier
    """
    grid_params = {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None], 'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10],
                   'n_estimators': [200, 600, 000, 1400, 1800, 2000]}
    gs = GridSearchCV(RandomForestClassifier(), grid_params, cv=5)
    return gs


def build_classifier(name : str ="kneighbors"):
    """Builds a supervised classifier belonging to one of multiple predefined types

    Keyword Arguments:
        name {str} -- Classifier's name (default: {"kneighbors"})

    Returns:
        Classifier Object -- Classifier object to be fitted on data
    """
    if (name == "kneighbors"):
        return build_knn()
    elif (name == "naive_bayes"):
        return build_NB()
    elif (name == "ensemble"):
        return build_ensemble()
