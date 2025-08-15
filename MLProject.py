##############################################
# Customer Churn Prediction ML App (Fully Commented)
# --------------------------------------------------
# Purpose:
# - Upload and explore a dataset
# - Preprocess: drop columns, impute, encode, scale
# - Train models: KNN, Random Forest, ANN (MLPClassifier)
# - Evaluate: metrics, confusion matrix, ROC/PR curves, feature importance
# - Predict on new inputs using trained artifacts
#
# Notes on techniques (TL;DR):
# - Label Encoding: convert categories to ints so ML models can use them.
# - Scaling: StandardScaler (mean=0, std=1) or MinMax (0..1) improves KNN/ANN performance
#             by making feature magnitudes comparable; RF is scale-invariant but keeping
#             a single pipeline simplifies use across models and prediction UI.
# - Stratified train/test split: keeps class ratio stable in both sets → fair evaluation.
# - Permutation Importance: model-agnostic feature importance based on performance drop
#             when shuffling a feature → more faithful than tree impurity alone.
##############################################

# =========================
# Imports
# =========================
import warnings
warnings.filterwarnings('ignore')  # Keep UI clean; does not affect computation

import io, json, pickle, zipfile, tempfile

import numpy as np
import pandas as pd

import streamlit as st

# Plotting libs (interactive + static)
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn: preprocessing, models, metrics, tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Track if user is on Home/Landing
if "show_home" not in st.session_state:
    st.session_state.show_home = True


# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="Customer Churn ML App", layout="wide")

# =========================
# Session State (keeps data/models across pages)
# =========================
if 'raw_data' not in st.session_state: st.session_state.raw_data = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'models' not in st.session_state: st.session_state.models = {}
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'label_encoders' not in st.session_state: st.session_state.label_encoders = {}
if 'feature_columns' not in st.session_state: st.session_state.feature_columns = []
if 'target_col' not in st.session_state: st.session_state.target_col = None
if 'feature_types' not in st.session_state: st.session_state.feature_types = {"numeric": [], "categorical": []}
if 'X_train' not in st.session_state: st.session_state.X_train = None
if 'X_test' not in st.session_state: st.session_state.X_test = None
if 'y_train' not in st.session_state: st.session_state.y_train = None
if 'y_test' not in st.session_state: st.session_state.y_test = None
if 'rf_importance' not in st.session_state: st.session_state.rf_importance = None
if 'perm_importance' not in st.session_state: st.session_state.perm_importance = {}  # {model_name: df}

# =========================
# Helper Functions
# =========================

def coerce_numeric_like_objects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object (string) columns that are actually numeric (e.g., '1,234 ') into numeric dtype.
    WHY: CSVs often misread numbers with commas/spaces as text; fixing improves stats/plots/models.
    """
    df_conv = df.copy()
    for col in df_conv.columns:
        if df_conv[col].dtype == 'object':
            try:
                df_conv[col] = pd.to_numeric(
                    df_conv[col].astype(str).str.replace(",", "").str.strip(),
                    errors='raise'
                )
            except Exception:
                # If conversion fails, leave as-is (it's truly categorical/text)
                pass
    return df_conv

def build_label_encoders(df: pd.DataFrame, categorical_cols: list) -> dict:
    """
    Fit & apply LabelEncoder to each categorical feature (not target here).
    WHY: Many sklearn models need numeric inputs; label encoding maps categories → integers.
         For ordinal issues: acceptable for tree models; for KNN/ANN, one-hot is often better,
         but here we keep a compact approach while ensuring scaling for numeric.
    """
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return encoders

def guess_target_name(df: pd.DataFrame) -> str:
    """
    Heuristic to suggest a likely target column by common names; else fallback: last column.
    """
    candidates = ['Churn', 'churn', 'Target', 'target', 'Exited', 'default', 'Label', 'label']
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]

def metrics_table(results: dict) -> pd.DataFrame:
    """
    Convert per-model results dict → tidy DataFrame for display.
    """
    return pd.DataFrame({
        name: [results[name]['Accuracy'], results[name]['Precision'],
               results[name]['Recall'], results[name]['F1-Score']]
        for name in results.keys()
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

def make_data_dictionary(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """
    Render a compact data dictionary including:
    - Column names, dtypes, example values, and whether numeric.
    WHY: Great for sanity checks before/after preprocessing.
    """
    st.markdown(f"#### {title}")
    example_vals = df.head(3).astype(str).agg(lambda s: ", ".join(list(s)[:3]))
    dd = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str),
        "Example Values": example_vals.values,
        "Is Numeric": df.dtypes.apply(lambda d: d.kind in "fiub").values
    })
    st.dataframe(dd, use_container_width=True)
    return dd

def plot_confusion_matrix(cm: np.ndarray, labels=None, title="Confusion Matrix"):
    """
    Draw a confusion matrix heatmap.
    WHY: Shows distribution of correct vs incorrect predictions per class.
    """
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

# =========================
# Page 1: Dataset & EDA
# =========================
def page1_dataset():
    st.title("📦 Dataset")
    st.caption("Upload your CSV, preview it, and explore with rich EDA.")

    # File uploader (CSV)
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        try:
            df = pd.read_csv(data_file)
            st.session_state.raw_data = df  # persist uploaded data
            st.success(f"File uploaded successfully! Shape: {df.shape}")

            # Show file metadata (nice UI touch)
            with st.expander("File details"):
                st.json({"file_name": data_file.name, "file_type": data_file.type, "file_size (bytes)": data_file.size})

            # --- Quick Preview & Column Info ---
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Preview")
                st.dataframe(df.head(50), use_container_width=True)
            with c2:
                st.subheader("Column Info")
                info_df = pd.DataFrame({
                    'Column': df.columns,
                    'Dtype': df.dtypes.astype(str),
                    'Non-Null': df.notnull().sum(),
                    'Null': df.isnull().sum()
                })
                st.dataframe(info_df, use_container_width=True)

            # --- EDA Widgets ---
            st.markdown("### 🔎 Exploratory Data Analysis")

            # Numeric distributions (hist+box): spot skew/outliers quickly
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                col = st.selectbox("Numeric distribution", num_cols, key="dist_num")
                fig = px.histogram(df, x=col, marginal="box", nbins=50, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

            # Categorical counts: class balance / category cardinality
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if cat_cols:
                col = st.selectbox("Categorical counts", cat_cols, key="dist_cat")
                counts_df = df[col].value_counts().reset_index()
                counts_df.columns = [col, 'count']
                fig = px.bar(counts_df, x=col, y='count', title=f"Counts of {col}")
                st.plotly_chart(fig, use_container_width=True)

            # Boxplot numeric by category: effect size / spread
            if num_cols and cat_cols:
                cnum, ccat = st.columns(2)
                with cnum:
                    num_for_box = st.selectbox("Boxplot: numeric", num_cols, key="box_num")
                with ccat:
                    cat_for_box = st.selectbox("Boxplot: by category", cat_cols, key="box_cat")
                fig = px.box(df, x=cat_for_box, y=num_for_box, title=f"{num_for_box} by {cat_for_box}")
                st.plotly_chart(fig, use_container_width=True)

            # Scatter matrix (sample): quick multivariate overview
            if len(num_cols) >= 2 and st.checkbox("Show scatter matrix (sampled to 1,000 rows)"):
                df_sample = df.sample(n=min(1000, len(df)), random_state=0)
                fig = px.scatter_matrix(df_sample[num_cols].iloc[:, :6], title="Scatter Matrix (first 6 numeric cols)")
                st.plotly_chart(fig, use_container_width=True)

            # Missingness map: pattern spotting (MNAR hints)
            if st.checkbox("Show missingness matrix"):
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
                ax.set_title("Missingness Matrix")
                st.pyplot(fig)

            # Correlation heatmap: multicollinearity quick check
            if len(num_cols) > 1 and st.checkbox("Show correlation heatmap"):
                corr = df[num_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax)
                ax.set_title("Correlation Heatmap (Numeric Features)")
                st.pyplot(fig)

            # Target-wise comparison if a likely target exists
            churn_col = None
            for c in ['Churn', 'churn', 'Target', 'target']:
                if c in df.columns:
                    churn_col = c; break
            if churn_col:
                st.markdown("#### Target-wise Comparison")
                col = st.selectbox("Pick a numeric feature to compare by target", num_cols or [churn_col], key="target_num")
                if col != churn_col and col in df.columns:
                    fig = px.violin(df, x=churn_col, y=col, box=True, points="all",
                                    title=f"{col} distribution by {churn_col}")
                    st.plotly_chart(fig, use_container_width=True)

            # Quick pivot: mean numeric by category
            st.markdown("#### Quick Pivot (Mean of Numeric by Category)")
            if num_cols and cat_cols:
                num_pivot = st.selectbox("Numeric value", num_cols, key="pivot_num")
                cat_pivot = st.selectbox("Category", cat_cols, key="pivot_cat")
                pivot_df = df.groupby(cat_pivot)[num_pivot].mean().reset_index()
                fig = px.bar(pivot_df, x=cat_pivot, y=num_pivot, title=f"Mean {num_pivot} by {cat_pivot}")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    else:
        st.info("Please upload a CSV to begin.")

# =========================
# Page 2: Preprocessing
# =========================
def page2_preprocessing():
    st.title("🛠️ Preprocessing")
    st.caption("Drop irrelevant columns, fix numeric types, choose target, impute, encode, scale. Includes data dictionary & processed download.")

    if st.session_state.raw_data is None:
        st.warning("Please upload a CSV on the Dataset page.")
        return

    # Work on a copy of raw data
    df = st.session_state.raw_data.copy()

    # Data dictionary BEFORE (helps confirm input schema)
    make_data_dictionary(df, "Data Dictionary (Before)")

    # --- 1) Drop unwanted columns (IDs etc.) ---
    st.subheader("1) Drop Unwanted Columns")
    drop_cols = st.multiselect(
        "Select columns to drop (IDs, free text, etc.)",
        df.columns.tolist(),
        default=[c for c in df.columns if c.lower() in ["customerid", "customer_id", "id", "customer id"]]
    )
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # --- 2) Fix numeric-looking text columns ---
    st.subheader("2) Fix Numeric Columns That Look Like Text")
    df = coerce_numeric_like_objects(df)
    st.info("Attempted to convert numeric-looking object columns to numeric.")

    # --- 3) Choose target column ---
    st.subheader("3) Select Target Column")
    suggested = guess_target_name(df)  # heuristic suggestion
    target_col = st.selectbox("Target variable", df.columns.tolist(),
                              index=df.columns.get_loc(suggested) if suggested in df.columns else 0)
    st.session_state.target_col = target_col

    # --- 4) Handle missing values ---
    st.subheader("4) Handle Missing Values")
    # WHY: Imputation retains rows instead of dropping them → preserves signal, avoids bias from listwise deletion.
    mv_col1, mv_col2 = st.columns(2)
    with mv_col1: num_strategy = st.selectbox("Numerical imputation", ["mean", "median", "most_frequent"], index=0)
    with mv_col2: cat_strategy = st.selectbox("Categorical imputation", ["most_frequent"], index=0)

    # Split features (excluding target) into numeric/categorical
    numeric_cols = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.drop(columns=[target_col]).select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        imputer_num = SimpleImputer(strategy=num_strategy)
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy=cat_strategy)
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # --- 5) Encode categoricals (features only) ---
    st.subheader("5) Encode Categorical Features")
    # WHY: Models need numerics; LabelEncoder is compact and quick.
    #      (Note: One-Hot often preferred for KNN/ANN, but here we keep a leaner pipeline.)
    label_encoders = {}
    if categorical_cols:
        label_encoders = build_label_encoders(df, categorical_cols)
        st.success(f"Encoded categorical features: {categorical_cols}")
    else:
        st.info("No categorical features detected.")

    # Encode target if it's categorical
    if df[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col].astype(str))
        label_encoders[target_col] = le_target
        st.info(f"Target '{target_col}' encoded as integers.")

    # --- 6) Feature Scaling ---
    st.subheader("6) Feature Scaling")
    # WHY Standardization?
    #   - KNN/ANN are distance/gradient-based; features on larger scales dominate otherwise.
    #   - StandardScaler: centers to mean=0, std=1 (good for Gaussian-ish data).
    #   - MinMaxScaler: squashes into [0,1] (useful when bounded inputs are preferred).
    scaling_method = st.selectbox("Select scaling method", ["None", "StandardScaler", "MinMaxScaler"], index=1)
    feature_cols = [c for c in df.columns if c != target_col]
    scaler = None
    if scaling_method != "None":
        scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        st.success(f"Scaled features with {scaling_method}.")

    # --- Save artifacts to session state ---
    st.session_state.processed_data = df
    st.session_state.label_encoders = label_encoders
    st.session_state.scaler = scaler
    st.session_state.feature_columns = feature_cols
    st.session_state.feature_types = {"numeric": numeric_cols, "categorical": categorical_cols}

    # Preview processed data + data dictionary AFTER
    with st.expander("Preview processed data"):
        st.dataframe(df.head(), use_container_width=True)
        st.info(f"Shape: {df.shape}")
    make_data_dictionary(df, "Data Dictionary (After)")

    # Download the processed CSV for reuse
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download Processed CSV",
        data=csv_buf.getvalue(),
        file_name="processed_data.csv",
        mime="text/csv"
    )

# =========================
# Page 3: Training
# =========================
def page3_training():
    st.title("🧠 Train Models (KNN • Random Forest • Artificial Neural Network)")
    st.caption("Split your data, pick algorithms, and train. ANN is included (MLPClassifier).")

    if st.session_state.processed_data is None or st.session_state.target_col is None:
        st.warning("Please complete Preprocessing first.")
        return

    df = st.session_state.processed_data
    target_col = st.session_state.target_col
    if target_col not in df.columns:
        st.error("Target column not found in processed data.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- 1) Train-Test Split ---
    st.subheader("1) Train-Test Split")
    # WHY stratify?
    #   - Preserves class distribution between train/test → avoids misleading metrics
    #   - test_size controls evaluation holdout fraction
    col1, col2 = st.columns(2)
    with col1: test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
    with col2: random_state = st.number_input("Random state", value=42, min_value=0, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    st.success(f"Split done. Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    st.session_state.X_train, st.session_state.X_test = X_train, X_test
    st.session_state.y_train, st.session_state.y_test = y_train, y_test

    # --- 2) Select & Configure Models ---
    st.subheader("2) Select & Configure Models")
    # WHY these 3?
    #   - KNN: simple, benefits strongly from scaling; good baseline for local structure.
    #   - Random Forest: robust, handles non-linearities, provides native feature importances.
    #   - ANN (MLP): flexible for complex patterns; needs scaling and more tuning.
    selected_models = st.multiselect(
        "Models to train",
        ["K-Nearest Neighbors", "Random Forest", "Artificial Neural Network"],
        default=["K-Nearest Neighbors", "Random Forest", "Artificial Neural Network"]
    )

    models = {}
    if "K-Nearest Neighbors" in selected_models:
        k_neighbors = st.slider("KNN: number of neighbors (k)", 3, 25, 7, 1)
        models["KNN"] = KNeighborsClassifier(n_neighbors=k_neighbors)

    if "Random Forest" in selected_models:
        n_estimators = st.slider("RF: number of trees", 50, 400, 200, 10)
        max_depth = st.slider("RF: max depth", 3, 40, 15, 1)
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    if "Artificial Neural Network" in selected_models:
        hidden_layers_text = st.text_input("ANN hidden layers (comma-separated)", "128,64")
        try:
            hidden_layers = tuple(int(x.strip()) for x in hidden_layers_text.split(",") if x.strip())
        except:
            hidden_layers = (128, 64)
        models["ANN"] = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=800, random_state=random_state)

    # --- Train ---
    if st.button("🚀 Train Selected Models", type="primary"):
        trained = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained[name] = model
                st.success(f"✅ {name} trained.")
            except Exception as e:
                st.error(f"❌ Error training {name}: {e}")

        st.session_state.models = trained

        # Random Forest native feature importance (Gini impurity decrease)
        if "Random Forest" in trained and hasattr(trained["Random Forest"], "feature_importances_"):
            rf = trained["Random Forest"]
            imp_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance": rf.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.session_state.rf_importance = imp_df

        # --- Bundle artifacts into a ZIP for download ---
        if trained:
            # WHY save artifacts?
            #   - Reuse models for batch scoring/production without retraining.
            artifacts = {
                "models": trained,
                "scaler": st.session_state.scaler,
                "label_encoders": st.session_state.label_encoders,
                "metadata": {
                    "target_col": target_col,
                    "feature_columns": st.session_state.feature_columns,
                    "feature_types": st.session_state.feature_types
                }
            }
            with tempfile.TemporaryFile() as tmp:
                with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
                    # models
                    for mname, mobj in trained.items():
                        z.writestr(f"models/{mname}.pkl", pickle.dumps(mobj))
                    # scaler (if used)
                    if st.session_state.scaler is not None:
                        z.writestr("scaler.pkl", pickle.dumps(st.session_state.scaler))
                    # label encoders (including target encoder if created)
                    z.writestr("label_encoders.pkl", pickle.dumps(st.session_state.label_encoders))
                    # metadata JSON
                    z.writestr("metadata.json", json.dumps(artifacts["metadata"], indent=2))
                tmp.seek(0)
                zip_bytes = tmp.read()
            st.download_button(
                "⬇️ Download Models & Artifacts (ZIP)",
                data=zip_bytes,
                file_name="models_and_artifacts.zip",
                mime="application/zip"
            )

# =========================
# Page 4: Evaluation
# =========================
def page4_evaluation():
    st.title("📈 Model Evaluation & Comparison")
    st.caption("Metrics, ROC & Precision-Recall curves, confusion matrices, and feature importances.")

    if not st.session_state.models:
        st.warning("Please train models first.")
        return

    models = st.session_state.models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # --- 1) Classification Metrics (per model) ---
    st.subheader("1) Classification Metrics")
    results = {}
    for name, model in models.items():
        # Predicted labels
        y_pred = model.predict(X_test)

        # Predicted probabilities (binary case only) for ROC/PR curves
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') and len(model.classes_) == 2 else None

        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            # Weighted averages handle class imbalance safely
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'y_pred': y_pred,
            'y_pred_proba': y_proba
        }

    mt = metrics_table(results).round(4)
    st.dataframe(mt, use_container_width=True)

    # Grouped bar chart of metrics
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=list(results.keys()), y=[results[n]['Accuracy'] for n in results.keys()]),
        go.Bar(name='Precision', x=list(results.keys()), y=[results[n]['Precision'] for n in results.keys()]),
        go.Bar(name='Recall', x=list(results.keys()), y=[results[n]['Recall'] for n in results.keys()]),
        go.Bar(name='F1-Score', x=list(results.keys()), y=[results[n]['F1-Score'] for n in results.keys()])
    ])
    fig.update_layout(title='Model Performance Comparison', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    best_model_name = mt.loc['Accuracy'].idxmax()
    st.success(f"🏆 Best model by Accuracy: **{best_model_name}**")

    # --- 2) Detailed Plots (Confusion Matrix, ROC, PR) ---
    st.subheader("2) Detailed Plots")
    selected_model = st.selectbox("Choose a model for plots", list(models.keys()))
    if selected_model:
        y_pred = results[selected_model]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels=getattr(models[selected_model], "classes_", None),
                              title=f"Confusion Matrix - {selected_model}")

        # ROC & PR only for binary problems with probabilities
        if results[selected_model]['y_pred_proba'] is not None:
            y_proba = results[selected_model]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
            fig1.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
            fig1.update_layout(title=f'ROC Curve - {selected_model}', xaxis_title='FPR', yaxis_title='TPR')
            st.plotly_chart(fig1, use_container_width=True)

            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'))
            fig2.update_layout(title=f'Precision-Recall Curve - {selected_model}',
                               xaxis_title='Recall', yaxis_title='Precision')
            st.plotly_chart(fig2, use_container_width=True)

    # --- 3) Feature Importances ---
    st.subheader("3) Feature Importances")
    feature_cols = st.session_state.feature_columns

    # Native RF importance (Gini)
    if st.session_state.rf_importance is not None:
        st.markdown("**Random Forest (Gini) Importances**")
        imp_df = st.session_state.rf_importance.copy()
        fig = px.bar(imp_df.head(15)[::-1], x="Importance", y="Feature", orientation="h",
                     title="Top Features (RF)")
        st.plotly_chart(fig, use_container_width=True)

    # Permutation importance: model-agnostic (slower)
    st.markdown("**Permutation Importance (model-agnostic)**")
    # WHY permutation importance?
    #   - Measures performance drop when a feature is randomly shuffled.
    #   - Works for any model; more faithful to predictive utility.
    if st.button("Compute Permutation Importance for All Models"):
        perm_results = {}
        for name, model in models.items():
            try:
                r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
                df_imp = pd.DataFrame({"Feature": feature_cols, "Importance": r.importances_mean}) \
                            .sort_values("Importance", ascending=False)
                perm_results[name] = df_imp
                st.session_state.perm_importance[name] = df_imp
                st.success(f"Computed permutation importance for {name}.")
            except Exception as e:
                st.error(f"Permutation importance failed for {name}: {e}")

    # Show stored permutation importances (if any)
    if st.session_state.perm_importance:
        model_for_perm = st.selectbox("View permutation importance for", list(st.session_state.perm_importance.keys()))
        df_imp = st.session_state.perm_importance[model_for_perm]
        fig = px.bar(df_imp.head(15)[::-1], x="Importance", y="Feature", orientation="h",
                     title=f"Permutation Importance - {model_for_perm}")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Page 5: Prediction
# =========================
def page5_prediction():
    st.title("🔮 Predict")
    st.caption("Enter unscaled values and original categories — the app will encode & scale for you.")

    if not st.session_state.models:
        st.warning("Please train a model first.")
        return
    if st.session_state.processed_data is None:
        st.warning("Please complete Preprocessing first.")
        return

    models = st.session_state.models
    target_col = st.session_state.target_col
    raw_df = st.session_state.raw_data           # original data (pre-preprocessing)
    proc_df = st.session_state.processed_data    # processed data (for fallback means)
    label_encoders = st.session_state.label_encoders
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_columns

    # Build input form using ORIGINAL column types (friendlier UX)
    raw_like_cols = [c for c in raw_df.columns if c in feature_cols]

    st.subheader("1) Enter Feature Values")
    input_data = {}
    cols_per_row = 3
    for i in range(0, len(raw_like_cols), cols_per_row):
        cols3 = st.columns(cols_per_row)
        for j, col_name in enumerate(raw_like_cols[i:i+cols_per_row]):
            series_raw = raw_df[col_name]

            # If the column was label-encoded during preprocessing, offer its original classes
            if col_name in label_encoders and col_name != target_col:
                classes = list(label_encoders[col_name].classes_)
                with cols3[j]:
                    input_val = st.selectbox(f"{col_name}", classes, key=f"inp_{col_name}")
                input_data[col_name] = input_val
            else:
                # Numeric input: set sensible default bounds from raw data distribution
                try:
                    s = pd.to_numeric(series_raw, errors='coerce')
                    min_val = float(np.nanmin(s)); max_val = float(np.nanmax(s)); mean_val = float(np.nanmean(s))
                except Exception:
                    # Fallback in case of parsing issues
                    min_val, max_val, mean_val = 0.0, 1e6, 0.0
                with cols3[j]:
                    input_data[col_name] = st.number_input(
                        f"{col_name}", value=mean_val,
                        min_value=min_val, max_value=max_val, key=f"inp_{col_name}"
                    )

    # Choose model + decision threshold (for binary proba models)
    st.subheader("2) Choose Model & Threshold")
    selected_model_name = st.selectbox("Model", list(models.keys()))
    threshold = st.slider("Decision Threshold (probability for positive class)", 0.0, 1.0, 0.50, 0.01)

    if st.button("Predict", type="primary"):
        # --- Assemble a single-row DF from form entries (raw values) ---
        input_df_raw = pd.DataFrame([input_data])

        # Encode categoricals exactly as during preprocessing (same encoders)
        input_df_encoded = input_df_raw.copy()
        for col, le in label_encoders.items():
            if col == target_col:
                continue  # never encode target here
            if col in input_df_encoded.columns:
                input_df_encoded[col] = le.transform(input_df_encoded[col].astype(str))

        # Align features to training columns; fill missing with processed means
        final_input = pd.DataFrame(columns=feature_cols)
        for c in feature_cols:
            if c in input_df_encoded.columns:
                final_input[c] = input_df_encoded[c]
            else:
                # If a feature wasn't in the form (rare), fallback to mean from processed data
                final_input[c] = proc_df[c].mean() if c in proc_df.columns else 0.0

        # Apply the SAME scaler learned during preprocessing
        if scaler is not None:
            final_input[feature_cols] = scaler.transform(final_input[feature_cols])

        # Predict using the chosen model
        model = models[selected_model_name]
        if hasattr(model, 'predict_proba') and len(getattr(model, "classes_", np.array([0,1]))) == 2:
            # Binary classification with probabilities → thresholding for positive class
            proba = model.predict_proba(final_input)[0]
            classes = getattr(model, "classes_", np.array([0,1]))
            pos_index = np.where(classes == classes.max())[0][0]  # assumes positive is numerically larger class
            pos_proba = proba[pos_index]
            pred_label = int(pos_proba >= threshold)
            confidence = pos_proba
        else:
            # Multiclass or models without proba: direct label prediction
            pred_label = int(model.predict(final_input)[0])
            confidence = None

        # --- Output UI ---
        st.subheader("3) Prediction Result")
        c1, c2 = st.columns(2)
        with c1:
            if pred_label == 1:
                st.error("Prediction: POSITIVE (e.g., CHURN)")
            else:
                st.success("Prediction: NEGATIVE (e.g., NO CHURN)")
        with c2:
            st.metric("Confidence", f"{confidence:.2%}" if confidence is not None else "N/A")

        # Probability breakdown (if available)
        if confidence is not None and hasattr(model, 'predict_proba'):
            prob_df = pd.DataFrame({'Class': [str(c) for c in getattr(model, "classes_", [0,1])],
                                    'Probability': model.predict_proba(final_input)[0]})
            st.bar_chart(prob_df.set_index("Class"))

        st.subheader("Input Summary")
        st.dataframe(pd.DataFrame(list(input_data.items()), columns=['Feature', 'Value']), use_container_width=True)

# =========================
# Page 6: Conclusions
# =========================
def page6_conclusions():
    st.title("🧭 Conclusions & Summary")
    st.caption("Interpret which features drove churn, how each model performed, and the trade-offs — with visual aids.")

    if st.session_state.processed_data is None or not st.session_state.models:
        st.info("Train some models first to see conclusions.")
        return

    models = st.session_state.models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    feature_cols = st.session_state.feature_columns

    # --- 1) Performance Summary (Test Set) ---
    st.subheader("1) Performance Summary (Test Set)")
    perf = {}
    for name, m in models.items():
        y_pred = m.predict(X_test)
        perf[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    perf_df = pd.DataFrame(perf).T.sort_values("Accuracy", ascending=False).round(4)
    st.dataframe(perf_df, use_container_width=True)

    fig = px.bar(perf_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score"),
                 x="index", y="Score", color="Metric", barmode="group", title="Model Metrics (Higher is Better)")
    fig.update_xaxes(title="Model"); fig.update_yaxes(title="Score")
    st.plotly_chart(fig, use_container_width=True)

    # --- 2) Feature importance overview ---
    st.subheader("2) Which Features Were Most Predictive of Churn?")
    # Prefer permutation importances if available; else fall back to RF Gini importance
    candidate_models = list(st.session_state.perm_importance.keys())
    use_perm = False
    if candidate_models:
        chosen = st.selectbox("Choose model for feature importance view",
                              candidate_models + (["Random Forest (Gini)"] if st.session_state.rf_importance is not None else []))
        if chosen in st.session_state.perm_importance:
            use_perm = True
            imp_df = st.session_state.perm_importance[chosen]
            title = f"Permutation Importance — {chosen}"
        else:
            imp_df = st.session_state.rf_importance
            title = "Random Forest (Gini) Importance"
    else:
        imp_df = st.session_state.rf_importance
        title = "Random Forest (Gini) Importance" if imp_df is not None else None

    if imp_df is not None and title is not None:
        fig = px.bar(imp_df.head(15)[::-1], x="Importance", y="Feature", orientation="h", title=title)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "- **Permutation importance**: performance drop when a feature is shuffled (model-agnostic; faithful to predictive signal).  \n"
            "- **Gini importance (RF)**: total impurity decrease across trees (fast, but model-specific)."
        )
    else:
        st.info("Feature importances not computed yet. Go to Evaluation → Compute Permutation Importance.")

    # --- 3) Interpretations & trade-offs ---
    st.subheader("3) Interpretation & Trade-offs")
    st.markdown("""
- **KNN**: thrives on well-scaled data; simple, but can be sensitive to noisy/irrelevant features and class imbalance.
- **Random Forest**: robust to non-linearities, good default, offers **feature importances** out of the box.
- **ANN (MLP)**: can capture complex relationships; needs scaling, tuning, and enough data.

**Trade-offs**
- Need **interpretability & stability** → start with **Random Forest**.
- Want **simplicity** and local pattern capture → try **KNN** (tune k; ensure scaling).
- Chasing **highest accuracy** on complex patterns and can tune → **ANN**.
""")

    # --- 4) Radar chart for quick visual comparison ---
    st.subheader("4) Visual Aid: Radar Comparison")
    md_norm = perf_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        if col in md_norm.columns:
            mmin, mmax = md_norm[col].min(), md_norm[col].max()
            md_norm[col] = 0.5 if mmax == mmin else (md_norm[col] - mmin) / (mmax - mmin)

    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig_radar = go.Figure()
    for idx, row in md_norm.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[c] for c in categories],
            theta=categories,
            fill='toself',
            name=idx
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                            showlegend=True, title="Normalized Metrics Radar")
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- 5) Practical Recommendations ---
    st.subheader("5) Practical Recommendations")
    st.markdown("""
1. Start with **Random Forest** as a strong, interpretable baseline.
2. Use **Permutation Importance** to confirm which features truly drive performance.
3. Tune **ANN** (layers, neurons, alpha, learning rate) if you need higher ceilings.
4. Adjust **decision threshold** (on Prediction page) to balance **precision vs recall** based on business costs.
5. Monitor drift: schedule periodic retraining and re-check importances.
""")

# =========================
# Sidebar Navigation
# =========================
def page_nav():
    # =========================
    # Sidebar Navigation
    # =========================

    # All page mappings here (outside function so they can be accessed anywhere)
    PAGES = {
        'Dataset': page1_dataset,
        'Preprocessing': page2_preprocessing,
        'Training': page3_training,
        'Evaluation': page4_evaluation,
        'Prediction': page5_prediction,
        'Conclusions': page6_conclusions
    }

    # Landing page (Home)
    if st.session_state.show_home:
        from pathlib import Path
        import base64

        # Path to the local image
        bg_image_path = Path(__file__).parent / "background.jpg"

        # Convert image to base64
        if bg_image_path.exists():
            with open(bg_image_path, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode()
        else:
            encoded_img = ""  # fallback if no file

        # CSS for background
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{encoded_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

        st.title("Welcome")
        st.markdown("""
        ### Predict Churn Easily
        - Upload your data & Explore with visualizations  
        - Train models  
        - Make predictions  
        """)

        if st.button("🌟 Get Started"):
            st.session_state.show_home = False
            st.rerun()
        st.stop()

    # Sidebar Navigation
    st.sidebar.title("Navigation")

    if st.sidebar.button("🏠 Home"):
        st.session_state.show_home = True
        st.rerun()

    selectpage = st.sidebar.radio("Go to", list(PAGES.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Group 6 Members:")
    st.sidebar.markdown("""
    1. Ekua Micah Abakah-Paintsil - 11012281  
    2. Jeremy Azumah - 11316332  
    3. Jonathan Tsekpo - 11352464  
    4. Adu-Twum Nana - 11123104  
    5. Asare Welbeck - 11014502
    """)

    # Render selected page
    PAGES[selectpage]()


# =========================
# Run App
# =========================
page_nav()