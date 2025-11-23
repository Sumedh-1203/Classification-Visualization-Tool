import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models import train_model
from plot_utils import plot_decision_boundary
from metrics_utils import compute_metrics

st.title("Classification Model Visualizer")

# -------------------------
# Session-state initialization
# -------------------------
if "active_dataset" not in st.session_state:
    # active_dataset is a dict: {"type": "builtin"/"advanced"/"custom", "name": str, "path": str (optional)}
    st.session_state["active_dataset"] = None

if "custom_df" not in st.session_state:
    st.session_state["custom_df"] = None

if "custom_cols" not in st.session_state:
    st.session_state["custom_cols"] = {"x1": None, "x2": None, "label": None}

if "adv_chosen" not in st.session_state:
    st.session_state["adv_chosen"] = None

# -------------------------
# Helper: small preview plotting
# -------------------------
def plot_preview_small(df, figsize=(2, 2), dotsize=8):
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(df["x1"], df["x2"], c=df["label"], s=dotsize)
    ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig)

# -------------------------
# Advanced dataset paths (single source of truth)
# -------------------------
ADV_PATHS = {
    "aggregation": "Adv_datasets/aggregation.csv",
    "compound": "Adv_datasets/compound.csv",
    "flame": "Adv_datasets/flame.csv",
    "jain": "Adv_datasets/jain.csv",
    "pathbased": "Adv_datasets/pathbased.csv",
    "spiral": "Adv_datasets/spiral.csv",
}

# -------------------------
# Sidebar: Dataset selector
# -------------------------
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["Dataset1", "Dataset2", "Explore Advanced Datasets", "Custom Dataset"],
    key="dataset_choice"
)

# If user selects builtin Dataset1/2, immediately set active_dataset and clear other dataset states
if dataset_choice in ["Dataset1", "Dataset2"]:
    st.session_state["active_dataset"] = {"type": "builtin", "name": dataset_choice, "path": dataset_choice}
    st.session_state["adv_chosen"] = None  # clear advanced
    # don't rerun — we can proceed

# -------------------------
# Advanced datasets flow
# -------------------------
if dataset_choice == "Explore Advanced Datasets":
    st.subheader("Explore Advanced Datasets")
    st.info("Datasets sourced from: https://www.kaggle.com/datasets/outsiders17711/2d-datasets")

    st.write("### Previews — click **Use <name>** under a plot to select that dataset")

    cols = st.columns(3)
    idx = 0
    # show all previews with a button under each
    for name, path in ADV_PATHS.items():
        df_prev = pd.read_csv(path)
        with cols[idx % 3]:
            st.write(f"**{name}**")
            plot_preview_small(df_prev, figsize=(2,2), dotsize=8)
            if st.button(f"Use {name}", key=f"use_adv_{name}"):
                # set active dataset in session state and rerun
                st.session_state["active_dataset"] = {"type": "advanced", "name": name, "path": path}
                st.session_state["adv_chosen"] = name
                st.rerun()
        idx += 1

    # If already chosen earlier, show a note and proceed (we don't stop)
    if st.session_state["adv_chosen"] is not None:
        st.success(f"Advanced dataset selected: **{st.session_state['adv_chosen']}** — continue to configure models or reset below.")

# -------------------------
# Custom dataset flow
# -------------------------
if dataset_choice == "Custom Dataset":
    st.sidebar.subheader("Upload Custom Dataset (CSV ≤ 5 MB)")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="custom_upload")

    # If a file is uploaded now, read and store to session state
    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.sidebar.error("File exceeds 5 MB limit.")
        else:
            try:
                df_tmp = pd.read_csv(uploaded_file)
                st.session_state["custom_df"] = df_tmp
                # reset chosen cols
                st.session_state["custom_cols"] = {"x1": None, "x2": None, "label": None}
            except Exception:
                st.sidebar.error("Could not read CSV — ensure it's a valid CSV.")

    # If there's no custom_df yet, prompt and stop (to avoid further code running)
    if st.session_state["custom_df"] is None:
        st.info("Upload a CSV file via the sidebar to use a custom dataset.")
    else:
        df_all = st.session_state["custom_df"]
        with st.expander("Preview uploaded file (first 5 rows)", expanded=True):
            st.write(df_all.head())

        st.sidebar.subheader("Select columns for x1, x2 and label")
        cols = list(df_all.columns)

        # pick defaults if previously set
        def_index = lambda val, options, fallback: options.index(val) if (val in options) else fallback

        default_x1 = st.session_state["custom_cols"].get("x1")
        default_x2 = st.session_state["custom_cols"].get("x2")
        default_label = st.session_state["custom_cols"].get("label")

        x1_col = st.sidebar.selectbox("x1 column", cols, index=def_index(default_x1, cols, 0), key="cust_x1")
        x2_col = st.sidebar.selectbox("x2 column", cols, index=def_index(default_x2, cols, 1 if len(cols)>1 else 0), key="cust_x2")
        label_col = st.sidebar.selectbox("label column", cols, index=def_index(default_label, cols, 2 if len(cols)>2 else 0), key="cust_label")

        # save choices
        st.session_state["custom_cols"] = {"x1": x1_col, "x2": x2_col, "label": label_col}

        # Show small preview scatter and a button to 'Use uploaded dataset'
        with st.expander("Preview and use uploaded dataset", expanded=True):
            try:
                preview_df = df_all[[x1_col, x2_col, label_col]].dropna()
                if preview_df.empty:
                    st.error("No rows after dropping NaN in selected columns.")
                else:
                    plot_preview_small(preview_df, figsize=(3,3))
                    if st.button("Use uploaded dataset", key="use_custom"):
                        # set active dataset and rerun
                        st.session_state["active_dataset"] = {"type": "custom", "name": "custom_uploaded", "path": None}
                        st.rerun()
            except Exception as e:
                st.error("Error creating preview — check selected columns.")

# -------------------------
# If no active dataset yet AND user picked builtin earlier, active_dataset will be set.
# If still no active dataset, prompt user to select one (to avoid crash)
# -------------------------
if st.session_state["active_dataset"] is None:
    st.warning("Please select a dataset (built-in, advanced or upload a custom CSV) from the sidebar or previews.")
    st.stop()

# -------------------------
# Load dataset according to active_dataset
# -------------------------
active = st.session_state["active_dataset"]
if active["type"] == "builtin":
    base = active["name"]
    df_train = pd.read_csv(f"{base}/train.csv")
    df_test  = pd.read_csv(f"{base}/test.csv")
    df_val   = pd.read_csv(f"{base}/val.csv")
    x1_col, x2_col, label_col = "x1", "x2", "label"

elif active["type"] == "advanced":
    path = active["path"]
    df_all = pd.read_csv(path)
    df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=42)
    df_val = df_test.copy()
    x1_col, x2_col, label_col = "x1", "x2", "label"

elif active["type"] == "custom":
    # use session custom_df and chosen cols
    df_all = st.session_state["custom_df"]
    cols = st.session_state["custom_cols"]
    x1_col, x2_col, label_col = cols["x1"], cols["x2"], cols["label"]
    df_all = df_all[[x1_col, x2_col, label_col]].dropna()
    if df_all.empty:
        st.error("Uploaded dataset has no valid rows after dropping NaN. Choose another dataset.")
        st.stop()
    df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=42)
    df_val = df_test.copy()

else:
    st.error("Unknown dataset type in session state.")
    st.stop()

# show which dataset is active and provide reset option
st.markdown(f"**Active dataset:** `{active['type']}` — **{active['name']}**")
if st.button("Reset dataset selection", key="reset_dataset"):
    st.session_state["active_dataset"] = None
    st.session_state["adv_chosen"] = None
    st.rerun()

# -------------------------
# Prepare arrays
# -------------------------
X_train = df_train[[x1_col, x2_col]].values
y_train = df_train[label_col].values
X_test  = df_test[[x1_col, x2_col]].values
y_test  = df_test[label_col].values
X_val   = df_val[[x1_col, x2_col]].values
y_val   = df_val[label_col].values

# -------------------------
# MODEL A settings (sidebar)
# -------------------------
st.sidebar.header("Model A – Settings")
methodA = st.sidebar.selectbox(
    "Choose Model A",
    ["KNN", "Logistic Regression", "MLFFNN-Based Classifier",
     "SVM (Linear Kernel)", "SVM (Polynomial Kernel)", "SVM (RBF Kernel)"],
    key="modelA"
)
paramsA: dict = {}
if methodA == "KNN":
    paramsA["k"] = st.sidebar.selectbox("K", [1,5,9], key="A_k")
elif methodA == "Logistic Regression":
    paramsA["degree"] = st.sidebar.selectbox("Degree", [1,3,5,7,9], key="A_deg")
elif methodA == "MLFFNN-Based Classifier":
    paramsA["layers"] = st.sidebar.selectbox("Hidden Layers", [1,2,3], key="A_layers")
    paramsA["neurons"] = st.sidebar.selectbox("Neurons per layer", [5,10,15,20], key="A_neurons")
    paramsA["eta"] = st.sidebar.selectbox("Learning Rate", [0.001,0.01,0.1,0.2], key="A_eta")
    paramsA["alpha"] = st.sidebar.selectbox("Momentum", [0.5,0.7,0.9], key="A_alpha")
elif methodA == "SVM (Linear Kernel)":
    paramsA["C"] = st.sidebar.selectbox("C", [1,10,100], key="A_C_lin")
elif methodA == "SVM (Polynomial Kernel)":
    paramsA["degree"] = st.sidebar.selectbox("Degree", [3,5,7,9], key="A_poly_deg")
    paramsA["C"] = st.sidebar.selectbox("C", [1,10,100], key="A_poly_C")
elif methodA == "SVM (RBF Kernel)":
    paramsA["gamma"] = st.sidebar.selectbox("Gamma", [2,0.5,0.125,0.02], key="A_gamma")
    paramsA["C"] = st.sidebar.selectbox("C", [1,10,100], key="A_rbf_C")

# -------------------------
# MODEL B (optional)
# -------------------------
compare = st.sidebar.checkbox("Compare with Model B?", key="compare")
paramsB: dict = {}
methodB: str | None = None
if compare:
    st.sidebar.header("Model B – Settings")
    methodB = st.sidebar.selectbox(
        "Choose Model B",
        ["KNN", "Logistic Regression", "MLFFNN-Based Classifier",
         "SVM (Linear Kernel)", "SVM (Polynomial Kernel)", "SVM (RBF Kernel)"],
        key="modelB"
    )
    if methodB == "KNN":
        paramsB["k"] = st.sidebar.selectbox("K (B)", [1,5,9], key="B_k")
    elif methodB == "Logistic Regression":
        paramsB["degree"] = st.sidebar.selectbox("Degree (B)", [1,3,5,7,9], key="B_deg")
    elif methodB == "MLFFNN-Based Classifier":
        paramsB["layers"] = st.sidebar.selectbox("Hidden Layers (B)", [1,2,3], key="B_layers")
        paramsB["neurons"] = st.sidebar.selectbox("Neurons per layer (B)", [5,10,15,20], key="B_neurons")
        paramsB["eta"] = st.sidebar.selectbox("Learning Rate (B)", [0.001,0.01,0.1,0.2], key="B_eta")
        paramsB["alpha"] = st.sidebar.selectbox("Momentum (B)", [0.5,0.7,0.9], key="B_alpha")
    elif methodB == "SVM (Linear Kernel)":
        paramsB["C"] = st.sidebar.selectbox("C (B)", [1,10,100], key="B_C_lin")
    elif methodB == "SVM (Polynomial Kernel)":
        paramsB["degree"] = st.sidebar.selectbox("Degree (B)", [3,5,7,9], key="B_poly_deg")
        paramsB["C"] = st.sidebar.selectbox("C (B)", [1,10,100], key="B_poly_C")
    elif methodB == "SVM (RBF Kernel)":
        paramsB["gamma"] = st.sidebar.selectbox("Gamma (B)", [2,0.5,0.125,0.02], key="B_gamma")
        paramsB["C"] = st.sidebar.selectbox("C (B)", [1,10,100], key="B_rbf_C")

# -------------------------
# Run training & show results
# -------------------------
if st.sidebar.button("Run Classification", key="run"):

    # Train model A
    modelA, XA_train, XA_test, XA_all, polyA, scalerA = train_model(
        methodA, paramsA, X_train, y_train, X_test, X_val
    )
    yA_train = modelA.predict(XA_train)
    yA_test = modelA.predict(XA_test)

    fmt = lambda x: f"{x*100:.4f}%"
    trainAccA = fmt((yA_train == y_train).mean())
    testAccA = fmt((yA_test == y_test).mean())

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Model A — Accuracy")
        st.table({
            "Metric": ["Train Accuracy", "Test Accuracy"],
            "Value": [trainAccA, testAccA]
        })
        st.subheader("Model A — Decision Boundary")
        st.pyplot(plot_decision_boundary(modelA, XA_train, y_train, X_train, yA_train, poly=polyA, scaler=scalerA))

    # Train & show model B if comparison enabled
    if compare and methodB is not None:
        modelB, XB_train, XB_test, XB_all, polyB, scalerB = train_model(
            methodB, paramsB, X_train, y_train, X_test, X_val
        )
        yB_train = modelB.predict(XB_train)
        yB_test = modelB.predict(XB_test)

        trainAccB = fmt((yB_train == y_train).mean())
        testAccB = fmt((yB_test == y_test).mean())

        with colB:
            st.subheader("Model B — Accuracy")
            st.table({
                "Metric": ["Train Accuracy", "Test Accuracy"],
                "Value": [trainAccB, testAccB]
            })
            st.subheader("Model B — Decision Boundary")
            st.pyplot(plot_decision_boundary(modelB, XB_train, y_train, X_train, yB_train, poly=polyB, scaler=scalerB))

        st.subheader("Model Comparison — Test Accuracy")
        st.table({
            "Metric": ["Model A Test Accuracy", "Model B Test Accuracy"],
            "Value": [testAccA, testAccB]
        })
