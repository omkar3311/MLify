# 🤖 **MLify**

✨ **MLify** is an interactive **machine learning & NLP web application** built using **Streamlit** that enables users to perform **end-to-end data science workflows** — from raw CSV data to trained models — without writing code.

🚀 **Live App:** https://mlifyy.streamlit.app/

---

![Demo ScreenShot](mlify.png)

---
## 🚀 **What MLify Can Do**

- 📂 Upload datasets in `.csv` format  
- 🔍 Perform automated **Exploratory Data Analysis (EDA)**  
- 📊 Generate **categorical, numerical & correlation visualizations**  
- 📈 Create **advanced custom plots** (scatter, bar, line, box, etc.)  
- 🛠️ Apply automatic **feature engineering**  
- 🎯 Detect task type (**classification or regression**)  
- 🤖 Train and compare multiple ML models  
- 🎛️ Perform **hyperparameter tuning** interactively  
- 📥 Download trained models as `.pkl` files  
- 🧠 Run **NLP pipelines** (supervised & unsupervised)  
- 🌀 Perform text clustering with **KMeans**  
- 🎨 Visualize clusters using **PCA**  
- ☁️ Generate **cluster-based WordClouds**  

---

## 🆕 Notebook-First Workflow (New)

MLify now **generates and exports a complete Jupyter Notebook** containing:

   - Data loading  
   - Cleaning & preprocessing 
   - EDA code 
   - Feature engineering   
   - Model training  
   - Evaluation metrics 

## 🤖 **Models Supported**

### 🧩 Classification
- 🌲 Random Forest  
- 📐 Logistic Regression  
- 🔀 Support Vector Machine (SVC)  

### 📈 Regression
- 📉 Linear Regression  
- 🚀 Gradient Boosting Regressor  

### 🧠 NLP (Supervised)
- 📐 Logistic Regression  
- 📊 Naive Bayes  
- ⚡ Linear SVM  

### 🌀 NLP (Unsupervised)
- 🔎 TF-IDF + KMeans Clustering  

---

## 📊 **Evaluation Metrics**

- ✅ Classification → Accuracy  
- 📏 Regression → R² Score  
- 🧭 NLP Clustering → Silhouette Score  

---

## 🎛️ **Hyperparameter Tuning**

### Classification
- Logistic Regression (C, max_iter)  
- Random Forest (n_estimators, max_depth)  
- SVM (C, kernel)  

### Regression
- Linear Regression  
- Gradient Boosting (n_estimators, learning_rate)  

**Includes:**
- Custom train–test split  
- Performance tracking  
- Model history  
- Download tuned models  

---

## 🧠 **NLP Module**

### 📘 Supervised NLP (Text Classification)
- TF-IDF vectorization  
- Logistic Regression  
- Naive Bayes  
- Linear SVM  
- Accuracy comparison  
- Best model export  

---

### 🌀 Unsupervised NLP (Clustering)
- TF-IDF + KMeans  
- Elbow method  
- Silhouette score  
- Automatic best-k selection  

---

### 🎨 NLP Visualization
- PCA-based 2D cluster plots  
- Cluster center visualization  

---

### ☁️ WordClouds
- Cluster-wise WordCloud generation  
- Frequency distribution charts  

---

## 🖥️ **Tech Stack**

- 🐍 Python  
- 🎈 Streamlit  
- 🧮 Pandas & NumPy  
- 📊 Matplotlib & Seaborn  
- 🤖 Scikit-learn  
- ☁️ WordCloud

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/omkar3311/MLify.git
cd automl-app
pip install -r requirements.txt
```
▶️ Usage
Run the Streamlit app:
```bash
streamlit run app.py
```
Open your browser at http://localhost:8501

## 👨‍💻 **Author**

   **Omkar Waghmare**  
🎓 Aspiring Data Scientist.
