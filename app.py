import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import io
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="MLify",page_icon="🤖",layout="wide")

if "page" not in st.session_state:
    st.session_state["page"] = "upload"

def next_page(p):
    st.session_state["page"] = p

def generate_plots(data):
    plots = {}
    category = data.select_dtypes(include='object').columns
    numerical = data.select_dtypes(include=['int64','float64']).columns
    for col in category:
        fig, ax = plt.subplots(2,1, figsize=(8, 8))
        counts = data[col].value_counts().head(20)
        sns.countplot(x=data[col], order=counts.index, ax=ax[0])
        ax[0].tick_params(axis='x', rotation=45)
        ax[0].set_title(f"Countplot: {col}")
        ax[1].pie(counts,labels=counts.index,autopct='%1.1f%%')
        ax[1].set_title(f"Pie Chart: {col}")
        plots[col] = fig
    for col in numerical:
        fig, ax = plt.subplots(2,1, figsize=(8, 8))
        sns.histplot(data[col], bins=10, kde=True, ax=ax[0])
        ax[0].set_title(f"Histogram: {col}")
        sns.boxplot(x=data[col], ax=ax[1])
        ax[1].set_title(f"Boxplot: {col}")
        plots[col] = fig
    if not numerical.empty:
        corr = data[numerical].corr()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plots["__heatmap__"] = fig
    return plots

def model_training(model, x, y, task="classification"):
    with st.spinner(f"Training {model.__class__.__name__}..."):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if task == "classification":
            score = int(accuracy_score(y_test, y_pred) * 100)
            st.metric(f"Accuracy for {model.__class__.__name__}", f"{score}%")
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            st.metric(f"RMSE for {model.__class__.__name__}", f"{rmse:.2f}")
            st.metric(f"R² for {model.__class__.__name__}", f"{r2:.2f}")
            score = r2
    return score

if st.session_state["page"] == "upload":
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 40vh;'>
            <h1 style='font-size: 80px; font-weight: bold; margin: 0;'>MLify</h1>
            <p style='font-size: 24px; color: gray; margin-top: 0px;'>
                Your Gateway to Smarter Data Insights
            </p>
        </div>
        """,
        unsafe_allow_html=True )
    file = st.file_uploader("Upload Your CSV File", type=["csv"])
    if file is not None and "data" not in st.session_state:
        with st.spinner("**Processing data...**"):
            st.session_state["data"] = pd.read_csv(file)
            st.session_state["summary"] = st.session_state["data"].describe().loc[['std','mean','count','50%']]
            st.session_state["data_cleaned"] = st.session_state["data"].drop_duplicates()
            st.session_state["plots"] = generate_plots(st.session_state["data_cleaned"])
        st.success("Processing complete ✅")
    if file is not None:
        data = st.session_state["data"]
        st.write("Here are the first few rows of your Data:")
        st.dataframe(data.head())
        target = st.selectbox("Choose Target Column", ['------Target------'] + data.columns.tolist(), key="target_column")
        if st.button("Submit"):
            if target == "------Target------":
                st.warning("Please select a valid target column before submitting.")
            else:
                st.session_state["target"] = target
                st.success(f"You selected **{target}** as the target column.")
                next_page("EDA_Summary")
                st.rerun()

elif st.session_state["page"] == "EDA_Summary":
    data = st.session_state["data"]
    col1,col2 = st.columns(2)
    with col1:
        col3,col4 = st.columns(2)
        with col3:
            st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: #f9f9f9;
                        text-align: center;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    ">
                    <h2 style="margin: 0; color: #333;">Rows</h2>
                    <h2 style="margin: 0; color: #333;">{data.shape[0]}</h2>
                    </div>
                    """,unsafe_allow_html=True)
        with col4:
            st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: #f9f9f9;
                        text-align: center;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    ">
                    <h2 style="margin: 0; color: #333;">Columns</h2>
                    <h2 style="margin: 0; color: #333;">{data.shape[1]}</h2>
                    </div>
                    """,unsafe_allow_html=True)
    with col2:
        data_cleaned = data.drop_duplicates()
        shape = data_cleaned.shape
        col3,col4 = st.columns(2)
        with col3:
            st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: #f9f9f9;
                        text-align: center;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    ">
                    <h2 style="margin: 0; color: #333;">Duplicates</h2>
                    <h2 style="margin: 0; color: #333;">{data_cleaned.shape[0]}</h2>
                    </div>
                    """,unsafe_allow_html=True)
            # st.metric("**Duplicates**",data_cleaned.shape[0])
        with col4:
            pass
    st.write("### Data Summary")
    # st.dataframe(st.session_state["summary"])
    # st.info(f"After removing duplicates: {st.session_state['data_cleaned'].shape[0]} rows remain")
    cards = 3  
    columns = list(data.select_dtypes(include=['int64','float64']).columns)
    for i in range(0, len(columns), cards):
        row_cols = st.columns(cards)
        for j, col_name in enumerate(columns[i:i + cards]):
            col_data = data[col_name]
            with row_cols[j]:
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: #f9f9f9;
                        text-align: center;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    ">
                        <h4 style="margin: 0; color: #333;">{col_name}</h4>
                        <p style="margin: 5px 0;"><strong>Missing:</strong> {col_data.isna().sum():.2f}</p>
                        <p style="margin: 5px 0;"><strong>Mean:</strong> {col_data.mean() }</p>
                        <p style="margin: 5px 0;"><strong>Median:</strong> {col_data.median() }</p>
                        <p style="margin: 5px 0;"><strong>Std Dev:</strong> {col_data.std() }</p>
                    </div>
                    """,
                    unsafe_allow_html=True )
    col1,col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        next_page("upload")
        st.rerun()
    if col2.button("➡️ Next"):
        next_page("visualizations")
        st.rerun()

elif st.session_state["page"] == "visualizations":
    st.write("### All Column Visualizations")
    data = st.session_state["data"].copy()
    categorical = data.select_dtypes(include='object').columns.tolist()
    numerical = data.select_dtypes(include=['int64','float64']).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        st.write("## Categorical Features")
        for col in categorical:
            st.write(f"#### {col}")
            st.pyplot(st.session_state["plots"][col])
    with col2:
        st.write("## Numerical Features")
        for col in numerical:
            st.write(f"#### {col}")
            st.pyplot(st.session_state["plots"][col])

    st.write("### Correlation Heatmap")
    st.pyplot(st.session_state["plots"]["__heatmap__"])
    col1,col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        next_page("EDA_Summary")
        st.rerun()
    if col2.button("➡️ Next"):
        next_page("training")
        st.rerun()

elif st.session_state["page"] == "training":
    data = st.session_state["data"].copy()
    target = st.session_state["target"]
    category = data.select_dtypes(include='object').columns
    label = LabelEncoder()
    for col in category:
        data[col] = label.fit_transform(data[col])
    x = data.drop(columns=[target])
    y = data[target]
    if y.dtype == "object" or len(y.unique()) < 20:
        task = "classification"
    else:
        task = "regression"
    st.write("### Model Training")
    if task == "classification":
        RF = model_training(RandomForestClassifier(n_estimators=5), x, y, task="classification")
        LR = model_training(LogisticRegression(max_iter=1000), x, y, task="classification")
        if RF > LR:
            best_model, best_model_name = RandomForestClassifier(n_estimators=5), "RandomForestClassifier"
        else:
            best_model, best_model_name = LogisticRegression(max_iter=1000), "LogisticRegression"
    else:
        score = model_training(LinearRegression(), x, y, task="regression")
        best_model, best_model_name = LinearRegression(), "LinearRegression"
    best_model.fit(x, y)
    st.session_state["best_model"] = best_model
    st.session_state["best_model_name"] = best_model_name
    col1,col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        next_page("visualizations")
        st.rerun()
    if col2.button("➡️ Next"):
        next_page("pkl")
        st.rerun()

elif st.session_state["page"] == "pkl":
    best_model = st.session_state["best_model"]
    best_model_name = st.session_state["best_model_name"]
    buffer = io.BytesIO()
    pickle.dump(best_model, buffer)
    buffer.seek(0)
    st.success(f"{best_model_name} model saved successfully!")
    st.download_button(
        label=f"Download {best_model_name} Model",
        data=buffer,
        file_name=f"{best_model_name}_model.pkl",
        mime="application/octet-stream")
    col1,col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        next_page("training")
        st.rerun()


