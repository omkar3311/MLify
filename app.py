import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import pickle
import io
import re
from sklearn.metrics import silhouette_score,accuracy_score, mean_squared_error, r2_score
import numpy as np
from services import generate_plots, adv_plot, unsupervised_graph, clouds,category_notebook, add_adv_plot_to_notebook , add_adv_model_to_notebook, generate_notebook_download,train_model,render_model_row
import nbformat 
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

st.set_page_config(page_title="MLify",page_icon="🤖",layout="wide")

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
    st.session_state["page"] = "home"
    
if "data" not in st.session_state:
    st.session_state["data"] = None
        
def next_page(p):
    st.session_state["page"] = p

if "adv_plot" not in st.session_state:
    st.session_state["adv_plot"] = {}
    
if "cells" not in st.session_state:
    st.session_state["cells"] = []

for k in ["import-data","shape","null","vis","encod","train","x&y","RF","LR","svc","GB"]:
    if k not in st.session_state:
        st.session_state[k] = False  

if "last_adv_plot" not in st.session_state:
    st.session_state["last_adv_plot"] = None

if "last_trained_model" not in st.session_state:
    st.session_state["last_trained_model"] = None

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
        
def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
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
                progress_value = max(0.0, min(1.0, r2))
                st.progress(progress_value)
            score = r2
    return score

def ui(model,score):
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown(f"""
                    <div  style = "color: #333;">
                    <h5>{model}</h5>
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

if st.session_state["page"] == "home":
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
    col1,col2 = st.columns(2)
    with col1:
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
                    <h2 style="margin: 0; color: #333;">Prediction</h2>
                    </div>
                    """,unsafe_allow_html=True)
        if st.button("➡️ Get started"):
            next_page("upload")
            st.rerun()
    with col2:
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
                    <h2 style="margin: 0; color: #333;">NLP</h2>
                    </div>
                    """,unsafe_allow_html=True)
         if st.button("➡️ Get started",key = "nlp"):
            next_page("nlp_home")
            st.rerun()
            
if st.session_state["page"] == "upload":
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 40vh;'>
            <h1 style='font-size: 80px; font-weight: bold; margin: 0;'>Regression/Classification</h1>
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
            if not st.session_state["import-data"]:
                st.session_state["cells"].append( new_markdown_cell("## Importing Data"))
                st.session_state["cells"].append( new_code_cell(
                        'import pandas as pd\n'
                        f'df = pd.read_csv("{file.name}")\n'
                        f'df.head()'))
                st.session_state["cells"].append( new_markdown_cell("## Summary"))
                st.session_state["cells"].append( new_code_cell(f'df.describe()'))
                st.session_state["import-data"] = True
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
    else:
        if st.button("⬅️ Back"):
            next_page("home")
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
        if not st.session_state["shape"]:
            st.session_state["cells"].append( new_markdown_cell("## Shape"))
            st.session_state["cells"].append( new_code_cell(f'df.shape'))
            st.session_state["shape"] = True
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
    if not st.session_state["null"]:
        st.session_state["cells"].append( new_markdown_cell("## Null Values"))
        st.session_state["cells"].append( new_code_cell(f'df.isna().sum()'))
        st.session_state["null"] = True
    
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

    if not st.session_state["vis"]:
        st.session_state["cells"].append( new_markdown_cell("## Visualization"))
        st.session_state["cells"].append( new_code_cell("""
import matplotlib.pyplot as plt\n
import seaborn as sns """))
        category_notebook()
        st.session_state["vis"] = True
    # new_pages("EDA_Summary","engg_feature")
    
elif st.session_state["page"] == "adv_visualization":
    page_title("Advanced Plots", "📊")
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
        if st.button("📈 Generate Plot"):
            if selected_plot == 'histplot':
                y = None
            adv_plot(selected_plot,data,x,y,hue)
            st.session_state["last_adv_plot"] = {
            "plot": selected_plot,
            "x": x,
            "y": y,
            "hue": hue }
        if "last_adv_plot" in st.session_state and st.session_state["last_adv_plot"] is not None:
            if st.button("➕ Add Plot to Notebook"):
                plot_info = st.session_state["last_adv_plot"]
                add_adv_plot_to_notebook(
                    plot_info["plot"],
                    plot_info["x"],
                    plot_info["y"],
                    plot_info["hue"] )
                

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
                            width: 400px;
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
                            width: 400px;
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
    cat = category.tolist()
    label = LabelEncoder()
    for col in category:
        data[col] = label.fit_transform(data[col])
    if not st.session_state["encod"]:
        st.session_state["cells"].append( new_markdown_cell("## Encode Categorical Features"))
        st.session_state["cells"].append( new_code_cell(
            f"""
from sklearn.preprocessing import LabelEncoder
df_encoded = df.copy()

label = LabelEncoder()
categorical_cols = {cat}

for col in categorical_cols:
    df_encoded[col] = label.fit_transform(df_encoded[col])

df_encoded.head()
        """.strip() ))
        st.session_state["encod"] = True
    x = data.drop(columns=[target])
    y = data[target]
    if not st.session_state["x&y"]:
        st.session_state["cells"].append( new_markdown_cell("## Feature and Target Selection"))
        st.session_state["cells"].append( new_code_cell(
            f"""
X = df_encoded.drop(columns=["{target}"])
y = df_encoded["{target}"]

X.shape, y.shape
        """.strip() ))
    
        st.session_state["cells"].append( new_markdown_cell("## Train Test Split"))
        st.session_state["cells"].append( new_code_cell(
            """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

X_train.shape, X_test.shape
    """.strip() ))
        st.session_state["x&y"] = True
    if y.dtype == "object" or len(y.unique()) < 20:
        task = "classification"
    else:
        task = "regression"
    results_container = st.container()
    with results_container:
        if task == "classification":
            RF = model_training(RandomForestClassifier(n_estimators=5), x, y, task="classification")
            ui("RandomForestClassifier",RF)
            if not st.session_state["RF"]:
                st.session_state["cells"].append( new_markdown_cell("## Random Forest Classifier"))
                st.session_state["cells"].append( new_code_cell(
                """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(n_estimators=5, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)
        """.strip() ))
                st.session_state["RF"] = True

            LR = model_training(LogisticRegression(max_iter=1000), x, y, task="classification")
            ui("LogisticRegression",LR)
            if not st.session_state["LR"]:
                st.session_state["cells"].append( new_markdown_cell("## Logistic Regression") )
                st.session_state["cells"].append(
                new_code_cell(
                """
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
accuracy_score(y_test, y_pred)
        """.strip() ))
                st.session_state["LR"] = True

            SV = model_training(SVC(C = 1,kernel='rbf'), x, y, task="classification")
            ui("SVC",SV)
            if not st.session_state["svc"] :
                st.session_state["cells"].append( new_markdown_cell("## Support Vector Classifier"))
                st.session_state["cells"].append(
                new_code_cell(
                """
from sklearn.svm import SVC
svc = SVC(C=1, kernel="rbf")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
accuracy_score(y_test, y_pred)
        """.strip() ))
                st.session_state["svc"] = True

        # if RF > LR and RF > SV:
        #     best_model, best_model_name = RandomForestClassifier(n_estimators=5), "RandomForestClassifier"
        # elif LR > SV:
        #     best_model, best_model_name = LogisticRegression(max_iter=1000), "LogisticRegression"
        # else:
        #     best_model, best_model_name = SVC(C = 1,kernel='rbf'), "SVC"
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=5, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVC": SVC(C=1, kernel="rbf") }
            
        else:
            # LR = model_training(LinearRegression(), x, y, task="regression")
            if not st.session_state["LR"]:
                st.session_state["cells"].append( new_markdown_cell("## Linear Regression"))
                st.session_state["cells"].append(
                new_code_cell(
                """
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
r2_score(y_test, y_pred)
        """.strip() ))
                st.session_state["LR"] = True

            # GB = model_training(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42), x, y, task="regression")
            if not st.session_state["GB"]:
                st.session_state["cells"].append( new_markdown_cell("## Gradient Boosting Regressor") )
                st.session_state["cells"].append( new_code_cell(
                """
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
r2_score(y_test, y_pred)
        """.strip()))
                st.session_state["GB"] = True
            models = {
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42 )}
        scores = {}

        for name, model in models.items():
            with st.spinner(f"Training {name}..."):
                score = train_model(model, x, y, task)
                scores[name] = score
                render_model_row(name, score, task)

        # if LR > GB :
        #     best_model, best_model_name = LinearRegression(), "LinearRegression"
        # else:
        #     best_model, best_model_name = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42), "GradientBoosting"
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]

    best_model.fit(x, y)
    st.session_state["best_model"] = best_model
    st.session_state["best_model_name"] = best_model_name

    # best_model.fit(x, y)
    
    # st.session_state["best_model"] = best_model
    # st.session_state["best_model_name"] = best_model_name
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
        label=f"⬇️ Download {best_model_name} Model",
        data=buffer,
        file_name=f"{best_model_name}_model.pkl",
        mime="application/octet-stream")
    col1,col2,col3,col4 = st.columns(4)
    if col1.button("⬅️ Back"):
        next_page("engg_feature")
        st.rerun()
    if col2.button("🔄 start Over"):
        st.session_state["cells"] = []
        for k in ["import-data","shape","null","vis","encod","train","x&y","RF","LR","svc","GB"]:
            if k in st.session_state:
                st.session_state[k] = False  
        next_page("upload")
        st.session_state["data"] = None
        st.rerun()
    if col3.button("🎛️ Hyperparameter tunning"):
        next_page("adv_training")
        st.rerun()
    nb_bytes = generate_notebook_download()
    if nb_bytes:
        col4.download_button(
            label="⬇️ Download Notebook",
            data=nb_bytes,
            file_name="model_notebook.ipynb",
            mime="application/x-ipynb+json"
        )
        
if "models" not in st.session_state:
    st.session_state.models = []
    
elif st.session_state["page"] == "adv_training":
    page_title("Model Tunning","🤖")
    data = st.session_state["data"].copy()
    target = st.session_state["target"]
    category = data.select_dtypes(include='object').columns
    label = LabelEncoder()
    for col in category:
        data[col] = label.fit_transform(data[col])
    x = data.drop(columns=[target])
    y = data[target]
    params = {}
    
    if y.dtype == "object" or len(y.unique()) < 20:
        task = "classification"
    else:
        task = "regression"
        
    if task == "classification":
        col1,col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox("Select Model",["Logistic Regression", "Random Forest", "SVM"])
        test_size = st.slider("Test Size (%)", 10, 50, 20, step=5) / 100
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        with st.spinner(f"{model_choice} Training..."):
            if model_choice == "Logistic Regression":
                params["C"] = st.number_input("Regularization Strength (C)", 0.01, 10.0, 1.0)
                params["max_iter"] = st.slider("Max Iterations", 100, 1000, 200)
                model = LogisticRegression(C=params["C"], max_iter=params["max_iter"])
            elif model_choice == "Random Forest":
                params["n_estimators"] = st.slider("Number of Trees", 10, 500, 100)
                params["max_depth"] = st.slider("Max Depth", 1, 20, 5)
                model = RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    random_state=42)
            elif model_choice == "SVM":
                params["C"] = st.number_input("Regularization (C)", 0.01, 10.0, 1.0)
                params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
                model = SVC(C=params["C"], kernel=params["kernel"], probability=True)
    else:
        col1,col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox("Select Model",["Linear Regression", "Gradient Boosting"])
        test_size = st.slider("Test Size (%)", 10, 50, 20, step=5) / 100
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        with st.spinner(f"{model_choice} Training..."):
            if model_choice == "Linear Regression":
                params["fit_intercept"] = True 
                model = LinearRegression()
            elif model_choice == "Gradient Boosting":
                params["n_estimators"] = st.slider("Number of Estimators", 50, 500, 100)
                params["learning_rate"] = round(st.number_input( "Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f" ), 2 )
                model = GradientBoostingRegressor(
                    n_estimators=params["n_estimators"],
                    learning_rate=params["learning_rate"],
                    random_state=42)
    if st.button("🛠️ Train Model"):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if model_choice in ["Logistic Regression", "Random Forest", "SVM"]:
            score = accuracy_score(y_test, y_pred)
            metric_name = "Accuracy"
        else:
            score = mean_squared_error(y_test, y_pred)
            metric_name = "MSE"
        st.success(f"{model_choice} trained! {metric_name}: {score:.4f}")
        st.session_state["last_trained_model"] = {
            "test_size": test_size,
            "model": model,
            "model_choice": model_choice,
            "metric_name": metric_name }

        pickle_file = f"{model_choice.replace(' ', '_').lower()}_model.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(model, f)
        st.session_state.models.append({
            "name": model_choice,
            "model": model,
            "score": score,
            "metric": metric_name,
            "file": pickle_file,
            "params": params
        })
    if "last_trained_model" in st.session_state:
        if st.button("➕ Add model in notebook"):
            info = st.session_state["last_trained_model"]
            add_adv_model_to_notebook(
                info["test_size"],
                info["model"],
                info["model_choice"],
                info["metric_name"] )

    if st.session_state.models:
        st.write("## Trained Models History")
        cols = st.columns(2)
        for i, entry in enumerate(st.session_state.models, 1):
            col = cols[(i - 1) % 2] 
            with col:
                st.write(f"### {i}. {entry['name']}")
                st.write(f"Performance: {entry['metric']} = {entry['score']:.4f}")
                if entry["params"]:
                    st.write("**Hyperparameters used:**")
                    st.json(entry["params"])
                with open(entry["file"], "rb") as f:
                    st.download_button(
                        label=f"Download {entry['name']} Model",
                        data=f,
                        file_name=entry["file"],
                        key=f"download_{i}")
                    
    col3,col4,col5 = st.columns(3)
    if col3.button("⬅️ Back"):
        next_page("training")
        st.rerun()
    if col4.button("🔄 start Over"):
        st.session_state["cells"] = []
        for k in ["import-data","shape","null","vis","encod","train","x&y","RF","LR","svc","GB"]:
            if k in st.session_state:
                st.session_state[k] = False 
        next_page("upload")
        st.session_state["data"] = None
        st.rerun()
        
    nb_bytes = generate_notebook_download()
    if nb_bytes:
        col5.download_button(
            label="⬇️ Download Notebook",
            data=nb_bytes,
            file_name="model_notebook.ipynb",
            mime="application/x-ipynb+json"
        )
        
elif st.session_state["page"] == "nlp_home":
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 40vh;'>
            <h1 style='font-size: 80px; font-weight: bold; margin: 0;'>NLP</h1>
            <p style='font-size: 24px; color: black; margin-top: 0px;'>
                Your Gateway to Smarter Data Insights
            </p>
        </div>
        """,
        unsafe_allow_html=True )
    file = st.file_uploader("📂 **Upload Your CSV File**", type=["csv"])
    if file:
            st.session_state["data"] = pd.read_csv(file)
            data = st.session_state["data"].copy()
            st.write("### 🔍 Data Preview")
            st.dataframe(data.head())
            st.session_state["text_col"] = st.selectbox("📝 Select Text Column", data.columns)
            st.session_state["target_col"] = st.selectbox("🎯 Select Target Column (or choose 'None' for clustering)", ["None"] + list(data.columns))
            if st.button("🚀 Run Analysis"):
                with st.spinner("🧹 Cleaning text..."):
                    data[st.session_state["text_col"]] = data[st.session_state["text_col"]].fillna("").apply(clean_text)
                vectorizer = TfidfVectorizer(max_features=5000)
                st.session_state["X"] = vectorizer.fit_transform(data[st.session_state["text_col"]])
                if st.session_state["target_col"] != "None":
                    next_page("supervised")
                    st.rerun()
                else:
                    next_page("unsupervised")
                    unsupervised_graph()
                    st.rerun()
    else:
        if st.button("⬅️ Back"):
            next_page("home")
            st.rerun()
elif st.session_state["page"] == "supervised":
            page_title("Supervised NLP", "🧠")
            data = st.session_state["data"].copy()
            st.subheader("📘 Supervised Learning (Text Classification)")
            y = data[st.session_state["target_col"]].astype(str)
            X_train, X_test, y_train, y_test = train_test_split(st.session_state["X"], y, test_size=0.25, random_state=42)
            models = {
                "Logistic Regression": LogisticRegression(max_iter=200),
                "Naive Bayes": MultinomialNB(),
                "Linear SVM": LinearSVC()}
            results = {}
            trained_models = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results[name] = acc
                trained_models[name] = model
            results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
            st.dataframe(results_df)
            plt.figure(figsize=(6,3))
            plt.bar(results_df["Model"], results_df["Accuracy"], color="teal")
            plt.title("Model Accuracy Comparison")
            plt.ylabel("Accuracy")
            st.pyplot(plt.gcf())
            plt.close()
            best_model_name = results_df.iloc[0]["Model"]
            best_model = trained_models[best_model_name]
            st.success(f"🏆 Best Model: {best_model_name} (Accuracy: {results_df.iloc[0]['Accuracy']:.4f})")
            buffer = io.BytesIO()
            pickle.dump(best_model, buffer)
            buffer.seek(0)
            col1,col2,col3 = st.columns(3)
            with col2:
                st.download_button(
                label=f"Download {best_model_name} Model",
                data=buffer,
                file_name=f"{best_model_name}_model.pkl",
                mime="application/octet-stream")
            if st.button("⬅️ Back"):
                next_page("nlp_home")
                st.rerun()
elif st.session_state["page"] == "unsupervised":
        page_title("Unsupervised NLP (Clustering)", "🌀")
        if "nlp_plots" in st.session_state and len(st.session_state["nlp_plots"]) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(st.session_state["nlp_plots"]["elbow"])
            with col2:
                st.pyplot(st.session_state["nlp_plots"]["silhouette"])
            st.success(f"✅ Best k (Silhouette Score): **{st.session_state['best_k']}**")
        col1,col2 = st.columns(2)
        with col1:
            new_pages("nlp_home","worldcloud")
        with col2:
            if st.button("➡️ Customize Graph"):
                next_page("nlp_graph")
                st.rerun()
elif st.session_state["page"] == "nlp_graph":
    page_title("NLP Cluster Visualization (PCA)", "🎨")
    user_k = st.slider("🎛️ Choose number of clusters (k)", 2, 10, st.session_state["best_k"])
    with st.spinner(f"Clustering with k={user_k}..."):
        X = st.session_state["X"]
        kmeans = KMeans(n_clusters=user_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        st.session_state["labels"] = labels
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X.toarray())
        centers_pca = pca.transform(kmeans.cluster_centers_)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.7)
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="X", s=200, c="red", label="Cluster Centers")
        ax.set_title(f"PCA Visualization of Clusters (k={user_k})")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)
        new_pages("unsupervised", "worldcloud")
                    
elif st.session_state["page"] == "worldcloud":
    page_title("Cluster WordClouds", "☁️")
    data = st.session_state["data"].copy()
    text_col = st.session_state["text_col"]
    X = st.session_state["X"]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    # labels = st.session_state["labels"]
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    st.bar_chart(cluster_counts)
    cluster_ids = sorted(cluster_counts.index.tolist())
    clouds(data,text_col,cluster_ids,labels)
    # for i in range(0, len(cluster_ids), 2):
    #     cols = st.columns(2)
    #     for j, col in enumerate(cols):
    #         if i + j < len(cluster_ids):
    #             cluster_id = cluster_ids[i + j]
    #             with col:
    #                 st.markdown(f"### Cluster {cluster_id}")
    #                 cluster_texts = " ".join(data[text_col][np.array(labels) == cluster_id])
    #                 wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cluster_texts)
    #                 fig, ax = plt.subplots(figsize=(6, 4))
    #                 ax.imshow(wordcloud, interpolation="bilinear")
    #                 ax.axis("off")
    #                 st.pyplot(fig)
    col1,col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        next_page("nlp_graph")
        st.rerun()
    if col2.button("🔄 start Over"):
        next_page("nlp_home")
        st.session_state["data"] = None
        st.rerun()