# Streamlit and UI-related
import streamlit as st
import time

# Core data processing
import pandas as pd
import numpy as np
import io

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Data type checks
import pandas.api.types as ptypes
from sklearn.utils.multiclass import type_of_target

# Model selection and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Model evaluation metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    accuracy_score, precision_score, recall_score, f1_score,  
    confusion_matrix, RocCurveDisplay  
)

# === Functions ===
def evaluate_regression_models(X, y):
    models = {
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        duration = end - start

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results.append({
            "Model": name,
            "Training Time (s)": f"{duration:.2f}",
            "MAE": f"{mae:.2f}",
            "MAPE (%)": f"{mape:.2f}",
            "MSE": f"{mse:.2f}",
            "RMSE": f"{rmse:.2f}",
            "R²": f"{r2:.2f}"
        })

    pred_vs_actual = pd.DataFrame({
            "Actual": y_test,
            "Predicted": preds
        }).reset_index(drop=True)

    st.write("🧾 **Predicted vs Actual Table**")
    st.dataframe(pred_vs_actual.head(10))
    results_df = pd.DataFrame(results)
    st.subheader("Regression Model Performance Summary")
    st.dataframe(results_df)


def evaluate_classification_models(X, y):
        models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            # "Logistic Regression": LogisticRegression(max_iter=1000),
            # "LGBM": LGBMClassifier()
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for name, model in models.items():
            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()
            duration = end - start

            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted')
            rec = recall_score(y_test, preds, average='weighted')
            f1 = f1_score(y_test, preds, average='weighted')

            
            st.subheader(name)
            st.write(f"**Training Time:** {duration:.2f} seconds")
            st.write(f"**Accuracy:** {acc:.2f}")
            st.write(f"**Precision:** {prec:.2f}")
            st.write(f"**Recall:** {rec:.2f}")
            st.write(f"**F1 Score:** {f1:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax_cm)
        ax_cm.set_title(f"Confusion Matrix: {name}")
        st.pyplot(fig_cm)

        # ROC Curve
        if probs is not None and len(np.unique(y_test)) == 2:
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, probs, ax=ax_roc, name=name)
            st.pyplot(fig_roc)

        st.markdown("---")


# === App Header ===
st.title("🤖 ML Suitability Checker – AutoML Assistant")
st.markdown("Upload a dataset and let this tool assess whether it's ready for machine learning tasks.")

st.divider()

uploaded_file = st.file_uploader("📂 Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully.")
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📋 Dataset Overview")

    with st.expander("ℹ️ Data Info (`df.info()`)", expanded=False):
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    st.write("📊 **Descriptive Statistics**")
    st.dataframe(df.describe())

    st.divider()

    all_columns = df.columns.tolist()
    y_col = st.selectbox("🎯 Select target (Y) column", all_columns)
    x_cols = st.multiselect("🧠 Select feature (X) columns", [col for col in all_columns if col != y_col])

    if y_col and x_cols:

        st.subheader("📈 Correlation with Target Variable")
        if y_col in df.columns and df[y_col].dtype in ['int64', 'float64']:
            corr = df.corr(numeric_only=True)
            if y_col in corr.columns:
                fig_corr, ax_corr = plt.subplots()
                corr[y_col].drop(y_col).sort_values().plot(kind='barh', ax=ax_corr)
                ax_corr.set_title(f'Correlation with {y_col}')
                st.pyplot(fig_corr)

        st.divider()

        st.subheader("🧪 ML Suitability Checks")

        if ptypes.is_numeric_dtype(df[y_col]):
            st.info("🔎 **Detected Problem Type:** Regression")
            problem_type = "regression"
        else:
            st.info("🔎 **Detected Problem Type:** Classification")
            problem_type = "classification"

        X = df[x_cols]
        y = df[y_col]

        if y.isnull().sum() > 0:
            st.warning(f"⚠️ Target column '{y_col}' contains {y.isnull().sum()} missing values. Dropping these rows...")
            initial_shape = df.shape
            df = df[~df[y_col].isnull()]
            X = df[x_cols]
            y = df[y_col]
            st.info(f"🧹 Dropped {initial_shape[0] - df.shape[0]} rows due to missing target values. Remaining: {df.shape[0]}")
        else:
            st.success(f"✅ No missing values detected in target column '{y_col}'.")

        x_null_rows = X.isnull().any(axis=1).sum()
        if x_null_rows > 0:
            st.warning(f"⚠️ Feature columns contain {x_null_rows} rows with missing values. Dropping them...")
            initial_shape = df.shape
            df = df.dropna(subset=x_cols)
            X = df[x_cols]
            y = df[y_col]
            st.info(f"🧹 Dropped {initial_shape[0] - df.shape[0]} rows due to feature nulls. Remaining: {df.shape[0]}")
        else:
            st.success("✅ No missing values detected in feature columns.")

        if not ptypes.is_numeric_dtype(y):
            st.write("🧠 Encoding target column...")
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.write(f"🔢 Encoded classes: {list(le.classes_)}")

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            st.write("🧠 Encoding categorical feature columns...")
            st.write(f"📌 Detected categorical columns: {cat_cols}")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        st.divider()

        # Run AutoML
        if problem_type == "regression":
            st.header("📉 Regression Model Performance Summary")
            evaluate_regression_models(X, y)

        elif problem_type == "classification":
            st.header("📊 Classification Model Performance Summary")
            evaluate_classification_models(X, y)

        st.divider()
        st.markdown("💡 *Tip: Improve data quality, tune models further, or export this notebook for extended work.*")
