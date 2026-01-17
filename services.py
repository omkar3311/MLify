import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,accuracy_score, mean_squared_error, r2_score

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
    
def unsupervised_graph():
    if "nlp_plots" not in st.session_state:
        st.session_state["nlp_plots"] = {}
    X = st.session_state["X"]
    k_values = range(2, 11)
    wcss, sil = [], []
    progress = st.progress(0)
    for idx, k in enumerate(k_values):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        sil.append(silhouette_score(X, labels))
        progress.progress((idx + 1) / len(k_values))
    progress.empty()
    fig_elbow, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(k_values, wcss, marker='o')
    ax1.set_title("📉 Elbow Method for Optimal k")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS")
    fig_sil, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(k_values, sil, marker='x', color='orange')
    ax2.set_title("📈 Silhouette Score vs k")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    st.session_state["nlp_plots"]["elbow"] = fig_elbow
    st.session_state["nlp_plots"]["silhouette"] = fig_sil
    best_k = k_values[np.argmax(sil)]
    st.session_state["best_k"] = best_k
    st.session_state["sil_scores"] = sil
    st.success(f"✅ Best k (Silhouette Score): **{best_k}** (Score = {max(sil):.4f})")
    
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