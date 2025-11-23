import streamlit as st
import pandas as pd
from models import train_model
from plot_utils import plot_decision_boundary
from metrics_utils import compute_metrics

st.title("Classification Model Visualizer")

# =========================================================
# DATASET SELECTION
# =========================================================
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["Dataset1", "Dataset2", "Custom Dataset"],
    key="dataset_choice"
)

# =========================================================
# LOAD BUILT-IN OR CUSTOM DATASET
# =========================================================

if dataset_choice in ["Dataset1", "Dataset2"]:
    data_path = dataset_choice

    df_train = pd.read_csv(f"{data_path}/train.csv")
    df_test  = pd.read_csv(f"{data_path}/test.csv")
    df_val   = pd.read_csv(f"{data_path}/val.csv")   # required for transforms even if not shown

    x1_col, x2_col, label_col = "x1", "x2", "label"

else:
    st.sidebar.subheader("Upload Custom Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (max 5 MB)",
        type=["csv"],
        key="custom_csv"
    )

    if uploaded_file is None:
        st.warning("Upload a CSV file to proceed.")
        st.stop()

    # Check file size (uploaded_file.size is in bytes)
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("File too large! Maximum allowed size is 5 MB.")
        st.stop()

    # Read CSV
    try:
        df_all = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Error reading CSV file. Ensure it is a valid CSV.")
        st.stop()

    # Column selection
    st.sidebar.subheader("Select Columns")
    x1_col = st.sidebar.selectbox("Select x1 column", df_all.columns, key="x1")
    x2_col = st.sidebar.selectbox("Select x2 column", df_all.columns, key="x2")
    label_col = st.sidebar.selectbox("Select output label column", df_all.columns, key="label")

    # Drop NaN rows
    df_all = df_all[[x1_col, x2_col, label_col]].dropna()

    if df_all.empty:
        st.error("Dataset has no valid rows after dropping NaN.")
        st.stop()

    # For custom dataset, train/test split 80-20
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=42)
    df_val = df_test.copy()   # placeholder, not used for accuracy display

# Extract arrays
X_train = df_train[[x1_col, x2_col]].values
y_train = df_train[label_col].values
X_test  = df_test[[x1_col, x2_col]].values
y_test  = df_test[label_col].values
X_val   = df_val[[x1_col, x2_col]].values
y_val   = df_val[label_col].values

# =========================================================
# SIDEBAR — MODEL A SETTINGS
# =========================================================
st.sidebar.header("Model A – Settings")

methodA = st.sidebar.selectbox(
    "Choose Model A",
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

    methodB = st.sidebar.selectbox(
        "Choose Model B",
        ["KNN",
         "Logistic Regression",
         "MLFFNN-Based Classifier",
         "SVM (Linear Kernel)",
         "SVM (Polynomial Kernel)",
         "SVM (RBF Kernel)"],
        key="modelB"
    )

    paramsB = {}

    # Hyperparams for Model B
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
    # MODEL A
    # =========================================================
    modelA, XA_train, XA_test, XA_all, polyA, scalerA = train_model(
        methodA, paramsA, X_train, y_train, X_test, X_val
    )

    y_pred_A_train = modelA.predict(XA_train)
    y_pred_A_test  = modelA.predict(XA_test)

    fmt = lambda x: f"{x*100:.4f}%"
    trainAccA = fmt((y_pred_A_train == y_train).mean())
    testAccA  = fmt((y_pred_A_test  == y_test).mean())

    # SIDE-BY-SIDE VIEW
    colA, colB = st.columns(2)

    # -------- MODEL A (LEFT) --------
    with colA:
        st.subheader("Model A — Accuracy")
        st.table({
            "Metric": ["Train Accuracy", "Test Accuracy"],
            "Value": [trainAccA, testAccA]
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

    # =========================================================
    # MODEL B
    # =========================================================
    if compare and methodB is not None:

        modelB, XB_train, XB_test, XB_all, polyB, scalerB = train_model(
            methodB, paramsB, X_train, y_train, X_test, X_val
        )

        y_pred_B_train = modelB.predict(XB_train)
        y_pred_B_test  = modelB.predict(XB_test)

        trainAccB = fmt((y_pred_B_train == y_train).mean())
        testAccB  = fmt((y_pred_B_test  == y_test).mean())

        with colB:
            st.subheader("Model B — Accuracy")
            st.table({
                "Metric": ["Train Accuracy", "Test Accuracy"],
                "Value": [trainAccB, testAccB]
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

        # FINAL COMPARISON
        st.subheader("Model Comparison — Test Accuracy")
        st.table({
            "Metric": ["Model A Test Accuracy", "Model B Test Accuracy"],
            "Value": [testAccA, testAccB]
        })
