import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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