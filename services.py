import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from sklearn.metrics import silhouette_score,accuracy_score, mean_squared_error, r2_score
import nbformat 
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

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

def category_notebook():
    category = st.session_state["data_cleaned"].select_dtypes(include="object").columns

    for col in category:
        st.session_state["cells"].append(
            new_code_cell(
                f"""
            fig, ax = plt.subplots(2, 1, figsize=(8, 15))

            counts = df["{col}"].value_counts().head(20)

            sns.countplot(
                x=df["{col}"],
                order=counts.index,
                ax=ax[0]
            )
            ax[0].tick_params(axis='x', rotation=45)
            ax[0].set_title("Countplot: {col}")

            ax[1].pie(
                counts,
                labels=counts.index,
                autopct='%1.1f%%'
            )
            ax[1].set_title("Pie Chart: {col}")

            plt.tight_layout()
            plt.show()
            """ ))

    numerical = st.session_state["data_cleaned"].select_dtypes(include=["int64", "float64"]).columns

    for col in numerical:
        st.session_state["cells"].append(
            new_code_cell(
                f"""
            fig, ax = plt.subplots(2, 1, figsize=(8, 15))

            sns.histplot(df["{col}"], bins=10, kde=True, ax=ax[0])
            ax[0].set_title("Histogram: {col}")

            sns.boxplot(x=df["{col}"], ax=ax[1])
            ax[1].set_title("Boxplot: {col}")

            plt.tight_layout()
            plt.show()
            """ ))
    if len(numerical) > 1:
        st.session_state["cells"].append(
            new_code_cell(
                f"""
                corr = df[{list(numerical)}].corr()

                plt.figure(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm")
                plt.title("Correlation Heatmap")
                plt.show()
                """ ))


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

def add_adv_plot_to_notebook(plot, x, y=None, hue=None):
    hue_code = f', hue="{hue}"' if hue else ""
    y_code = f', y="{y}"' if y else ""

    plot_code = f"""
        plt.figure(figsize=(8, 6))
        sns.{plot}(data=df, x="{x}"{y_code}{hue_code}{', kde=True' if plot == 'histplot' else ''})
        plt.title("{plot.title()} - {x}")
        plt.tight_layout()
        plt.show()
        """
    st.session_state["cells"].append( new_markdown_cell( f"### {plot.title()} ({x}{', ' + y if y else ''})"))
    st.session_state["cells"].append( new_code_cell(plot_code.strip()))


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

def add_adv_model_to_notebook(test_size,model,model_choice,metric_name):
    st.session_state["cells"].append(new_markdown_cell(f"##{model_choice}"))
    metric_code = (
        "accuracy_score(y_test, y_pred)"
        if metric_name == "Accuracy"
        else "mean_squared_error(y_test, y_pred)" )
    st.session_state["cells"].append(new_code_cell(f""" 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
            model = {model}
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = {metric_code}
            score   """  ))
    

def clouds(data,text_col,cluster_ids,labels):
    plots = {}
    for i in range(0, len(cluster_ids), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(cluster_ids):
                cluster_id = cluster_ids[i + j]
                with col:
                    st.markdown(f"### Cluster {cluster_id}")
                    cluster_texts = " ".join(data[text_col][np.array(labels) == cluster_id])
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cluster_texts)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plots[i] = fig
    return plots

