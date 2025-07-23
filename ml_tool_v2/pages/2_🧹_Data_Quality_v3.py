import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.utils.multiclass import type_of_target
import pandas.api.types as ptypes
import io
import json



st.set_page_config(page_title="Data Quality Checker", layout="wide")
st.title("🧹 Data Quality Check Tool")

# -------------------------------
# 🗂️ Upload and Preprocessing
# -------------------------------
st.header("📂 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Supported formats: CSV, Excel (.xls/.xlsx), JSON, TXT",
    type=["csv", "xls", "xlsx", "json", "txt"]
)

st.divider()

# Threshold Settings
st.header("⚙️ Configuration")
col1, col2 = st.columns(2)
with col1:
    missing_threshold = st.slider(
        "🔍 Missing data threshold (%)",
        min_value=0, max_value=100, value=30, step=5
    )
with col2:
    dominance_threshold = st.slider(
        "🧮 Dominance threshold (%) – flag columns where one value dominates",
        min_value=50, max_value=100, value=95, step=1
    )

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        elif file_type == "json":
            df = pd.read_json(uploaded_file)
        elif file_type == "txt":
            df = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("❌ Unsupported file type.")
            df = None

        if 'df' in locals():
            st.success("✅ File uploaded and read successfully!")
            st.subheader("👀 Preview of the Data")
            st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")

    # -------------------------------
    # 📊 Dataset Overview
    # -------------------------------
    st.divider()
    st.header("📊 Dataset Overview")
    st.write(f"**Rows:** {df.shape[0]} &nbsp;&nbsp;&nbsp; **Columns:** {df.shape[1]}")

    st.subheader("📋 Data Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe())

    # -------------------------------
    # 🧩 Completeness Check
    # -------------------------------
    st.divider()
    st.header("🧩 Completeness Check (Missing Values)")

    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    completeness_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_percent.round(2)
    }).sort_values(by="Missing %", ascending=False)

    st.dataframe(completeness_df)

    high_missing_cols = completeness_df[completeness_df["Missing %"] > missing_threshold]
    if not high_missing_cols.empty:
        st.warning(f"⚠️ Columns with more than {missing_threshold}% missing data:")
        st.dataframe(high_missing_cols)
    else:
        st.success(f"✅ No columns exceed the {missing_threshold}% missing threshold.")

    # -------------------------------
    # 🛠️ Accuracy Check
    # -------------------------------
    st.divider()
    st.header("🛠️ Accuracy Check (Data Types & Anomalies)")

    st.markdown("**📄 Data Types:**")
    st.dataframe(df.dtypes.rename("Type"))

    potential_issues = []
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col].astype(float)
            potential_issues.append(col)
        except:
            continue

    if potential_issues:
        st.warning("🔍 These object columns may contain numeric values:")
        st.write(potential_issues)
    else:
        st.success("✅ No suspicious object-numeric columns detected.")

    for col in potential_issues:
        with st.expander(f"🔎 Sample non-numeric values in `{col}`"):
            non_numeric = df[~df[col].str.replace('.', '', 1).str.isnumeric()][col].unique()
            st.write(non_numeric[:5])

    # -------------------------------
    # 🔁 Consistency Check
    # -------------------------------
    st.divider()
    st.header("🔁 Consistency Check (Duplicate Rows)")

    duplicate_count = df.duplicated().sum()
    st.write(f"🔄 Total duplicate rows: **{duplicate_count}**")

    if duplicate_count > 0:
        with st.expander("🪞 Show duplicated rows"):
            st.dataframe(df[df.duplicated()])
    else:
        st.success("✅ No duplicate rows found.")

    # -------------------------------
    # 🎯 Relevance Check
    # -------------------------------
    st.divider()
    st.header("🎯 Relevance Check")

    st.markdown("**🧍 Columns with only one unique value:**")
    single_unique_cols = [col for col in df.columns if df[col].nunique() == 1]
    if single_unique_cols:
        st.warning("These columns have only one unique value:")
        st.write(single_unique_cols)
    else:
        st.success("✅ No single-unique-value columns found.")

    st.markdown(f"**📌 Columns with a dominant value (>{dominance_threshold}%):**")
    dominance = {}
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
        if top_freq > (dominance_threshold / 100):
            dominance[col] = top_freq

    if dominance:
        dom_df = pd.DataFrame.from_dict(dominance, orient='index', columns=['Dominance %'])
        dom_df['Dominance %'] = (dom_df['Dominance %'] * 100).round(2)
        st.dataframe(dom_df)
    else:
        st.success(f"✅ No columns exceed the {dominance_threshold}% dominance threshold.")

    # -------------------------------
    # 📉 Correlation Heatmap
    # -------------------------------
    st.divider()
    st.header("📉 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("ℹ️ Not enough numerical columns for a heatmap.")

    # -------------------------------
    # ⚖️ Bias Check
    # -------------------------------
    st.divider()
    st.header("⚖️ Bias Check (Distributions & Outliers)")

    st.subheader("🧮 Categorical Value Distributions")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        st.markdown(f"**{col}**")
        st.bar_chart(df[col].value_counts())
        top_val_ratio = df[col].value_counts(normalize=True).values[0]
        if top_val_ratio > 0.95:
            st.warning(f"⚠️ Column `{col}` is highly imbalanced: Top value = {round(top_val_ratio * 100, 2)}%")

    st.subheader("📊 Numerical Distributions & Outliers")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if num_cols:
        fig, axes = plt.subplots(nrows=len(num_cols), ncols=2, figsize=(12, 4 * len(num_cols)))
        for i, col in enumerate(num_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f"{col} - Histogram")
            axes[i, 0].set_xlabel("")

            sns.boxplot(x=df[col], ax=axes[i, 1])
            axes[i, 1].set_title(f"{col} - Boxplot")
            axes[i, 1].set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ No numeric columns to plot.")
else:
    st.info("📥 Please upload a file to begin the analysis.")
