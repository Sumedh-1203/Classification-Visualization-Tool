from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np

def train_model(method, params, X_train, y_train, X_test, X_val):

    model = None
    scaler = None
    poly = None

    # Combined for decision boundary
    X_all = np.vstack([X_train, X_test, X_val])

    # Default (no transform)
    X_train_processed = X_train
    X_test_processed  = X_test
    X_all_processed   = X_all

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------
    if method == "KNN":
        model = KNeighborsClassifier(n_neighbors=params["k"])

    # --------------------------------------------------------
    # LOGISTIC REGRESSION WITH POLYNOMIAL FEATURES
    # --------------------------------------------------------
    elif method == "Logistic Regression":
        poly = PolynomialFeatures(degree=params["degree"])
        X_train_processed = poly.fit_transform(X_train)
        X_test_processed  = poly.transform(X_test)
        X_all_processed   = poly.transform(X_all)
        model = LogisticRegression(max_iter=500)

    # --------------------------------------------------------
    # MLFFNN
    # --------------------------------------------------------
    elif method == "MLFFNN-Based Classifier":
        layers = tuple([params["neurons"]] * params["layers"])
        scaler = StandardScaler()

        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed  = scaler.transform(X_test)
        X_all_processed   = scaler.transform(X_all)

        model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation='tanh',
            solver='sgd',
            learning_rate_init=params["eta"],
            momentum=params["alpha"],
            max_iter=1000
        )

    # --------------------------------------------------------
    # SVM LINEAR
    # --------------------------------------------------------
    elif method == "SVM (Linear Kernel)":
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed  = scaler.transform(X_test)
        X_all_processed   = scaler.transform(X_all)
        model = SVC(kernel="linear", C=params["C"], probability=True)

    # --------------------------------------------------------
    # SVM POLYNOMIAL
    # --------------------------------------------------------
    elif method == "SVM (Polynomial Kernel)":
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed  = scaler.transform(X_test)
        X_all_processed   = scaler.transform(X_all)
        model = SVC(kernel="poly",
                    degree=params["degree"],
                    C=params["C"],
                    probability=True)

    # --------------------------------------------------------
    # SVM RBF
    # --------------------------------------------------------
    elif method == "SVM (RBF Kernel)":
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed  = scaler.transform(X_test)
        X_all_processed   = scaler.transform(X_all)
        model = SVC(kernel="rbf",
                    gamma=params["gamma"],
                    C=params["C"],
                    probability=True)

    # Safety check
    if model is None:
        raise ValueError(f"Unknown method: {method}")

    # Train
    model.fit(X_train_processed, y_train)

    return model, X_train_processed, X_test_processed, X_all_processed, poly, scaler
