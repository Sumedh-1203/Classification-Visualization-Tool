import streamlit as st
import pandas as pd
from models import train_model
from plot_utils import plot_decision_boundary
from metrics_utils import compute_metrics

st.title("Classification Model Visualizer")

# =========================================================
# LOAD DATA
# =========================================================
df_train = pd.read_csv("Dataset2/train.csv")
df_val   = pd.read_csv("Dataset2/val.csv")
df_test  = pd.read_csv("Dataset2/test.csv")

X_train = df_train[['x1','x2']].values
y_train = df_train['label'].values
X_val   = df_val[['x1','x2']].values
y_val   = df_val['label'].values
X_test  = df_test[['x1','x2']].values
y_test  = df_test['label'].values

# =========================================================
# SIDEBAR — MODEL A SETTINGS
# =========================================================
st.sidebar.header("Model A – Settings")

methodA = st.sidebar.selectbox("Choose Model A",
    ["KNN",
     "Logistic Regression",
     "MLFFNN-Based Classifier",
     "SVM (Linear Kernel)",
     "SVM (Polynomial Kernel)",
     "SVM (RBF Kernel)"],
     key="modelA"
)

paramsA: dict = {}

# ---------------- Hyperparameters for Model A ----------------
if methodA == "KNN":
    paramsA["k"] = st.sidebar.selectbox("K", [1, 5, 9], key="A_k")

elif methodA == "Logistic Regression":
    paramsA["degree"] = st.sidebar.selectbox("Degree", [1,3,5,7,9], key="A_deg")

elif methodA == "MLFFNN-Based Classifier":
    paramsA["layers"] = st.sidebar.selectbox("Hidden Layers", [1,2,3], key="A_layers")
    paramsA["neurons"] = st.sidebar.selectbox("Neurons per layer", [5,10,15,20], key="A_neurons")
    paramsA["eta"] = st.sidebar.selectbox("Learning Rate", [0.001,0.01,0.1,0.2], key="A_eta")
    paramsA["alpha"] = st.sidebar.selectbox("Momentum", [0.5,0.7,0.9], key="A_alpha")

elif methodA == "SVM (Linear Kernel)":
    paramsA["C"] = st.sidebar.selectbox("C", [1,10,100], key="A_C")

elif methodA == "SVM (Polynomial Kernel)":
    paramsA["degree"] = st.sidebar.selectbox("Degree", [3,5,7,9], key="A_poly_deg")
    paramsA["C"] = st.sidebar.selectbox("C", [1,10,100], key="A_poly_C")

elif methodA == "SVM (RBF Kernel)":
    paramsA["gamma"] = st.sidebar.selectbox("Gamma", [2, 0.5, 0.125, 0.02], key="A_rbf_gamma")
    paramsA["C"] = st.sidebar.selectbox("C", [1,10,100], key="A_rbf_C")


# =========================================================
# SIDEBAR — MODEL B (OPTIONAL)
# =========================================================
methodB: str | None = None
paramsB: dict = {}

compare = st.sidebar.checkbox("Compare with Model B?", key="compare")

if compare:
    st.sidebar.header("Model B – Settings")

    methodB = st.sidebar.selectbox("Choose Model B",
        ["KNN",
         "Logistic Regression",
         "MLFFNN-Based Classifier",
         "SVM (Linear Kernel)",
         "SVM (Polynomial Kernel)",
         "SVM (RBF Kernel)"],
         key="modelB"
    )

    paramsB = {}

    # ---------------- Hyperparameters for Model B ----------------
    if methodB == "KNN":
        paramsB["k"] = st.sidebar.selectbox("K (Model B)", [1,5,9], key="B_k")

    elif methodB == "Logistic Regression":
        paramsB["degree"] = st.sidebar.selectbox("Degree (Model B)", [1,3,5,7,9], key="B_deg")

    elif methodB == "MLFFNN-Based Classifier":
        paramsB["layers"] = st.sidebar.selectbox("Hidden Layers (Model B)", [1,2,3], key="B_layers")
        paramsB["neurons"] = st.sidebar.selectbox("Neurons per layer (Model B)", [5,10,15,20], key="B_neurons")
        paramsB["eta"] = st.sidebar.selectbox("Learning Rate (Model B)", [0.001,0.01,0.1,0.2], key="B_eta")
        paramsB["alpha"] = st.sidebar.selectbox("Momentum (Model B)", [0.5,0.7,0.9], key="B_alpha")

    elif methodB == "SVM (Linear Kernel)":
        paramsB["C"] = st.sidebar.selectbox("C (Model B)", [1,10,100], key="B_C")

    elif methodB == "SVM (Polynomial Kernel)":
        paramsB["degree"] = st.sidebar.selectbox("Degree (Model B)", [3,5,7,9], key="B_poly_deg")
        paramsB["C"] = st.sidebar.selectbox("C (Model B)", [1,10,100], key="B_poly_C")

    elif methodB == "SVM (RBF Kernel)":
        paramsB["gamma"] = st.sidebar.selectbox("Gamma (Model B)", [2,0.5,0.125,0.02], key="B_rbf_gamma")
        paramsB["C"] = st.sidebar.selectbox("C (Model B)", [1,10,100], key="B_rbf_C")


# =========================================================
# RUN TRAINING
# =========================================================
if st.sidebar.button("Run Classification", key="run"):
    
    # =========================================================
    # MODEL A — TRAIN
    # =========================================================
    modelA, XA_train, XA_test, XA_all, polyA, scalerA = train_model(
        methodA, paramsA, X_train, y_train, X_test, X_val
    )

    y_pred_A_train = modelA.predict(XA_train)
    y_pred_A_test  = modelA.predict(XA_test)

    fmt = lambda x: f"{x*100:.4f}%"

    trainA = compute_metrics(y_train, y_pred_A_train)
    testA  = compute_metrics(y_test,  y_pred_A_test)

    # =========================================================
    # SIDE-BY-SIDE DISPLAY: MODEL A & MODEL B
    # =========================================================
    colA, colB = st.columns(2)

    # ------------- MODEL A COLUMN -------------
    with colA:
        st.subheader("Model A — Metrics")
        st.table({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Train": [
                fmt(trainA["Accuracy"]),
                fmt(trainA["Precision"]),
                fmt(trainA["Recall"]),
                fmt(trainA["F1 Score"])
            ],
            "Test": [
                fmt(testA["Accuracy"]),
                fmt(testA["Precision"]),
                fmt(testA["Recall"]),
                fmt(testA["F1 Score"])
            ]
        })

        st.subheader("Model A — Decision Boundary")
        st.pyplot(plot_decision_boundary(
            modelA,
            XA_train,
            y_train,
            X_train,
            y_pred_A_train,
            poly=polyA,
            scaler=scalerA
        ))

    # ------------- MODEL B COLUMN (IF ENABLED) -------------
    if compare and methodB is not None:
        
        # Train Model B
        modelB, XB_train, XB_test, XB_all, polyB, scalerB = train_model(
            methodB, paramsB, X_train, y_train, X_test, X_val
        )

        y_pred_B_train = modelB.predict(XB_train)
        y_pred_B_test  = modelB.predict(XB_test)

        trainB = compute_metrics(y_train, y_pred_B_train)
        testB  = compute_metrics(y_test,  y_pred_B_test)

        with colB:
            st.subheader("Model B — Metrics")
            st.table({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Train": [
                    fmt(trainB["Accuracy"]),
                    fmt(trainB["Precision"]),
                    fmt(trainB["Recall"]),
                    fmt(trainB["F1 Score"])
                ],
                "Test": [
                    fmt(testB["Accuracy"]),
                    fmt(testB["Precision"]),
                    fmt(testB["Recall"]),
                    fmt(testB["F1 Score"])
                ]
            })

            st.subheader("Model B — Decision Boundary")
            st.pyplot(plot_decision_boundary(
                modelB,
                XB_train,
                y_train,
                X_train,
                y_pred_B_train,
                poly=polyB,
                scaler=scalerB
            ))

        # ------------- COMPARISON TABLE -------------
        st.subheader("Model Comparison — Test Set")
        st.table({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Model A": [
                fmt(testA["Accuracy"]),
                fmt(testA["Precision"]),
                fmt(testA["Recall"]),
                fmt(testA["F1 Score"])
            ],
            "Model B": [
                fmt(testB["Accuracy"]),
                fmt(testB["Precision"]),
                fmt(testB["Recall"]),
                fmt(testB["F1 Score"])
            ]
        })
