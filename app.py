import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier , GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import io
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="MLify",page_icon="🤖",layout="wide")

import streamlit as st

st.markdown(
    """
    <style>
    .css-18e3th9 { 
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        margin: 0;
    }
    body {
        background: linear-gradient(to bottom, #6a0dad, #e6e6fa);
        background-attachment: fixed;
        margin: 0;
        height: 100vh;
        width: 100%;
    }
    .stApp, .css-1d391kg {
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state["page"] = "upload"

def next_page(p):
    st.session_state["page"] = p

def generate_plots(data):
    plots = {}
    category = data.select_dtypes(include='object').columns
    numerical = data.select_dtypes(include=['int64','float64']).columns
    for col in category:
        fig, ax = plt.subplots(2,1, figsize=(8, 15))
        counts = data[col].value_counts().head(20)
        sns.countplot(x=data[col], order=counts.index, ax=ax[0])
        ax[0].tick_params(axis='x', rotation=45)
        ax[0].set_title(f"Countplot: {col}")
        ax[1].pie(counts,labels=counts.index,autopct='%1.1f%%')
        ax[1].set_title(f"Pie Chart: {col}")
        plots[col] = fig
    for col in numerical:
        fig, ax = plt.subplots(2,1, figsize=(8, 15))
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
if "adv_plot" not in st.session_state:
    st.session_state["adv_plot"] = {}
def adv_plot(selected_plot, data, x, y=None, hue=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if selected_plot == "scatterplot":
        sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax)
    elif selected_plot == "lineplot":
        sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax)
    elif selected_plot == "histplot":
        sns.histplot(data=data, x=x, hue=hue, ax=ax, kde=True)
    elif selected_plot == "barplot":
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax)
    elif selected_plot == "boxplot":
        sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax)
    st.pyplot(fig)
    plot_key = f"{selected_plot}_{x}_{y if y else 'None'}"
    st.session_state["adv_plot"][plot_key] = fig
def model_training(model, x, y, task="classification"):
    with st.spinner(f"Training {model.__class__.__name__}..."):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if task == "classification":
            col1,col2,col3 = st.columns(3)
            with col1:
                score = int(accuracy_score(y_test, y_pred) * 100)
                st.markdown(f"""
                            <div  style = "color: #333;">
                            <h5>{model.__class__.__name__}</h5>
                            <hr>
                        """,unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                            <div  style = "color: #333;">
                            <h5>{score}%</h5>
                            <hr>
                        """,unsafe_allow_html=True)
            with col3:
                st.progress(score)
            
        else:
            # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div style="color: #333; text-align:center;">
                        <h5>{model.__class__.__name__}</h5>
                        <hr>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div style="color: #333; text-align:center;">
                        <h5>R²: {r2:.2f}</h5>
                        <hr>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.progress(r2)
            score = r2
    return score
def page_title(title,emoji=""):
    st.markdown(f"""
            <div style = "
                text-align: center;
                margin-top : 0px;
                margin-bottom : 40px;
                ">
                <h1>{emoji} {title}</h1>
                </div>
            """,unsafe_allow_html=True )
def new_pages(back,next):
    col1,col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        next_page(back)
        st.rerun()
    if col2.button("➡️ Next"):
        next_page(next)
        st.rerun()

if st.session_state["page"] == "upload":
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 40vh;'>
            <h1 style='font-size: 80px; font-weight: bold; margin: 0;'>🤖 MLify</h1>
            <p style='font-size: 24px; color: black; margin-top: 0px;'>
                Your Gateway to Smarter Data Insights
            </p>
        </div>
        """,
        unsafe_allow_html=True )
    file = st.file_uploader("**Upload Your CSV File 📂**", type=["csv"])
    if file is not None :
        with st.spinner("**Processing data...**"):
            st.session_state["data"] = pd.read_csv(file)
            st.session_state["summary"] = st.session_state["data"].describe().loc[['std','mean','count','50%']]
        with st.spinner("**Generating plots...**"):
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
    page_title("EDA", "📊")
    data = st.session_state["data"]
    col1,col2 = st.columns(2)
    with col1:
        col3,col4 = st.columns(2)
        with col3:
            st.markdown(
                    f"""
                    <div style="
                        border: 2px solid black;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: transparent;
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
                        border: 2px solid black;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: transparent;
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
                        border: 2px solid black;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: transparent;
                        text-align: center;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    ">
                    <h2 style="margin: 0; color: #333;">Duplicates</h2>
                    <h2 style="margin: 0; color: #333;">{data_cleaned.shape[0]}</h2>
                    </div>
                    """,unsafe_allow_html=True)
        with col4:
            pass
    st.write("### Data Summary")
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
                        border: 2px solid black;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: transparent;
                        text-align: center;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    ">
                        <h4 style="margin: 0; color: #333;">{col_name}</h4>
                        <p style="margin: 5px 0;color: #333;"><strong>Missing:</strong> {col_data.isna().sum():.2f}</p>
                        <p style="margin: 5px 0;color: #333;"><strong>Mean:</strong> {col_data.mean() }</p>
                        <p style="margin: 5px 0;color: #333;"><strong>Median:</strong> {col_data.median() }</p>
                        <p style="margin: 5px 0;color: #333;"><strong>Std Dev:</strong> {col_data.std() }</p>
                    </div>
                    """,
                    unsafe_allow_html=True )
    new_pages("upload","visualizations")

elif st.session_state["page"] == "visualizations":
    page_title("Visualizations", "📈")
    data = st.session_state["data"].copy()
    categorical = data.select_dtypes(include='object').columns.tolist()
    numerical = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.write("## Categorical Features")
    cards = 2
    for i in range(0, len(categorical), cards):
        row_cols = st.columns(cards)
        for j, col_name in enumerate(categorical[i:i + cards]):
            with row_cols[j]:
                
                    st.markdown(
                        f"""
                        <div style="
                            border: 2px solid black;
                            border-radius: 8px;
                            padding: 15px;
                            margin-bottom: 15px;
                            background-color: transparent;
                            text-align: center;
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                        ">
                            <h4 style="margin: 0; color: #333;">{col_name}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True)
                    st.pyplot(st.session_state["plots"][col_name]) 
    st.write("## Numerical Features")
    cards = 2
    for i in range(0, len(numerical), cards):
        row_cols = st.columns(cards)
        for j, col_name in enumerate(numerical[i:i + cards]):
            with row_cols[j]:
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid black;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 15px;
                        background-color: transparent;
                        text-align: center;
                        color: #333;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    ">
                        <h4 style="margin: 0; color: #333;">{col_name}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.pyplot(st.session_state["plots"][col_name])
    st.write("### Correlation Heatmap")
    st.pyplot(st.session_state["plots"]["__heatmap__"])
    col1,col2,col3 = st.columns(3)
    if col1.button("⬅️ Back"):
        next_page("EDA_Summary")
        st.rerun()
    if col2.button("➡️ Next"):
        next_page("engg_feature")
        st.rerun()
    if col3.button("➡️ Advance Visualization"):
        next_page("adv_visualization")
        st.rerun()
    # new_pages("EDA_Summary","engg_feature")
elif st.session_state["page"] == "adv_visualization":
    data = st.session_state["data"].copy()
    seaborn = ["scatterplot","lineplot","histplot","barplot","boxplot",]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='margin-bottom:0.2rem; color:#333;'>Choose Plot Type</h4>", unsafe_allow_html=True)
        selected_plot = st.selectbox(" ", seaborn, label_visibility="collapsed")

        st.markdown("<h4 style='margin-bottom:0.2rem; color:#333;'>Choose Hue</h4>", unsafe_allow_html=True)
        hue = st.selectbox(" ", [None] + list(data.columns), label_visibility="collapsed")  
    with col2:
        st.markdown("<h4 style='margin-bottom:0.2rem; color:#333;'>Choose X-axis</h4>", unsafe_allow_html=True)
        x = st.selectbox(" ", data.columns, label_visibility="collapsed")

        if selected_plot in ["scatterplot", "lineplot", "barplot", "boxplot"]:
            st.markdown("<h4 style='margin-bottom:0.2rem; color:#333;'>Choose Y-axis</h4>", unsafe_allow_html=True)
            y_cols = [col for col in data.columns if col != x]
            y = st.selectbox(" ", y_cols, label_visibility="collapsed")
    col3,col4 = st.columns(2)
    with col4:
        if st.button("Generate Plot"):
            adv_plot(selected_plot,data,x,y,hue)
    if len(st.session_state["plots"]) > 1:
        st.subheader("📜 Previous Plots")
        plots_list = list(st.session_state["adv_plot"].items())
        cards = 2
        for i in range(0, len(plots_list), cards):
            row_cols = st.columns(cards)
            for j, (plot_name, fig) in enumerate(plots_list[i:i + cards]):
                with row_cols[j]:
                    st.markdown(
                        f"""
                        <div style="
                            border: 2px solid black;
                            border-radius: 8px;
                            padding: 15px;
                            margin-bottom: 15px;
                            background-color: transparent;
                            text-align: center;
                            color: #333;
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                        ">
                            <h4 style="margin: 0; color: #333;">{plot_name}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.pyplot(fig)
    new_pages("visualizations","engg_feature")
elif st.session_state["page"] == "engg_feature":
    data = st.session_state["data"].copy()
    categorical = data.select_dtypes(include='object').columns.tolist()
    numerical = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_html = "<hr>".join(categorical)
    numerical_html = "<hr>".join(numerical)
    page_title("Feature Engineering", "🛠️")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown(
                f""" <div style="
                            border: 2px solid black;
                            border-radius: 8px;
                            padding: 15px;
                            background-color: transparent;
                            text-align: center;
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                            width: 500px;
                            margin-left : 60px;
                            color: #333;
                            margin-right : 10px;
                        ">
                            <h2>Categorical Columns</h2>
                            <hr style = "color: #333;">
                            <h4 >{categorical_html}</h4>
                        </div>
                        </div>
                """,
                unsafe_allow_html=True )
    with col2:
        st.markdown(
                f""" <div style="
                            border: 2px solid black;
                            border-radius: 8px;
                            padding: 15px;
                            background-color: transparent;
                            text-align: center;
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                            width: 500px;
                            color: #333;
                            margin-left : 10px;
                        ">
                            <h2>Numerical Columns</h2>
                            <hr>
                            <h4>{numerical_html}</h4>
                        </div>
                        </div>
                """,
                unsafe_allow_html=True )
    st.markdown(f"""
                    <div style = "
                    height:20px;
                    border: 0px solid #ddd;
                    border-radius: 8px;
                    " >  </div>
                    """,unsafe_allow_html=True )
    new_pages("visualizations","training")

elif st.session_state["page"] == "training":
    page_title("Model Training", "🤖")
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
    if task == "classification":
        RF = model_training(RandomForestClassifier(n_estimators=5), x, y, task="classification")
        LR = model_training(LogisticRegression(max_iter=1000), x, y, task="classification")
        SV = model_training(SVC(C = 1,kernel='rbf'), x, y, task="classification")
        if RF > LR and RF > SV:
            best_model, best_model_name = RandomForestClassifier(n_estimators=5), "RandomForestClassifier"
        elif LR > SV:
            best_model, best_model_name = LogisticRegression(max_iter=1000), "LogisticRegression"
        else:
            best_model, best_model_name = SVC(C = 1,kernel='rbf'), "SVC"
    else:
        LR = model_training(LinearRegression(), x, y, task="regression")
        GB = model_training(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42), x, y, task="regression")
        if LR > GB :
            best_model, best_model_name = LinearRegression(), "LinearRegression"
        else:
            best_model, best_model_name = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42), "GradientBoosting"
    best_model.fit(x, y)
    st.session_state["best_model"] = best_model
    st.session_state["best_model_name"] = best_model_name
    buffer = io.BytesIO()
    pickle.dump(best_model, buffer)
    buffer.seek(0)
    st.markdown(f"""
            <div style= "
                border: 2px solid black;
                            border-radius: 8px;
                            padding: 15px;
                            background-color: #90EE90;
                            text-align: center;
                            margin-left : 25%;
                            margin-bottom:30px;
                            width : 60%; 
                            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                "><h4>✅ {best_model_name} model saved successfully!</h4>
            </div>
        """,unsafe_allow_html=True )
    col1,col2,col3 = st.columns(3)
    with col2:
        st.download_button(
        label=f"Download {best_model_name} Model",
        data=buffer,
        file_name=f"{best_model_name}_model.pkl",
        mime="application/octet-stream")
    col1,col2 = st.columns(2)
    col1,col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        next_page("engg_feature")
        st.rerun()
    if col2.button("🔄 start Over"):
        next_page("upload")
        st.session_state["data"] = None
        st.rerun()