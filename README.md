# 🏡 House Price Prediction App

## 📊 Model Evaluation & Analysis
This project includes a comprehensive data science workflow, focusing on high-accuracy price estimation and model transparency:
* *Visualizations:* Feature correlation analysis using *Heatmaps*.
* *Classification Metrics:* Performance evaluation via *Confusion Matrix* (for binned price categories).
* *Regression Metrics:* Model accuracy measured using *MAE, **MSE, and **RMSE*.
* *Statistical Fit:* Model variance explained using the *R² Score*.

## 📂 Project Structure
* *app.py*: Interactive Streamlit web interface.
* *train.py*: Script for data cleaning, scaling, and model training.
* *Housepriceprediction.ipynb*: Research notebook with all EDA and graphs.
* *house_model.pkl*: Serialized Linear Regression model.
* *scaler.pkl*: Saved StandardScaler object for input normalization.
* *USA_Housing.csv*: Raw dataset used for training.
* *requirements.txt*: List of necessary Python libraries.

## 🛠️ Tech Stack
* *Language:* Python
* *ML Libraries:* Scikit-Learn, Pandas, NumPy
* *Visualization:* Seaborn, Matplotlib
* *Deployment:* Streamlit

## 🚀 Setup Instructions
1. Install dependencies:
   pip install -r requirements.txt
2. Run the application:
   streamlit run app.py
