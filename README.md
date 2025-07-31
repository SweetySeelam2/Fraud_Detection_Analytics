[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetection-analytics.streamlit.app/)

---

# 🔐 Enterprise-Grade Credit Card Fraud Detection System


An advanced, interactive, enterprise-grade fraud detection platform built with **XGBoost**, **SHAP**, and **Streamlit**. This solution identifies fraudulent credit card transactions in real-time with explainability and actionable business insights.

---

## 📌 Project Overview

Credit card fraud costs businesses billions every year. This app tackles that challenge by:
- Predicting fraud using a trained **XGBoost classifier**
- Providing **SHAP + LIME explainability**
- Enabling **manual data upload or sampling**
- Offering **clear business ROI insights** for decision-makers

---

## 🚀 Features

- ✅ Real-time fraud prediction
- ✅ Upload your own CSV or test on demo samples
- ✅ Explainability using SHAP and LIME
- ✅ Business savings estimates per detection
- ✅ Multi-page Streamlit app for easy navigation

---

## 📁 Streamlit App Pages

### 🏠 Home  
Short description of the app, model, features.

### 📁 Upload/Test Data  
Upload your own CSV with `Time`, `V1–V28`, and `Amount` columns or use a balanced demo sample (1500 rows, includes both fraud and non-fraud cases for realistic testing).

### 🤖 Predict Fraud  
Predict fraudulent transactions, see probability scores, and download results.

### 📊 Explainability  
Visualize model behavior using:
- SHAP summary plots
- SHAP force plot for individual transactions
- LIME HTML report

### 📈 Business Insights  
Summarizes:
- 💰 Estimated savings ($500 saved per fraud caught)
- 📊 Detection rate
- 📎 Strategic recommendations

---

## 📊 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: ~285K transactions | 492 frauds | PCA-transformed features

⚠️ Due to GitHub size limits, the full dataset (`creditcard.csv`) is downloaded directly from **Google Drive** via script:

gdown.download(DRIVE_URL, CSV_FILE)

A separate, balanced sample dataset (test_demo.csv) is generated for unbiased demo/testing in the app (contains a representative mix of fraud and non-fraud cases).
---

## 💡 Business Impact

- 🎯 **Fraud detection rate**: ~99% on test sample
- 💵 **Avg savings per fraud caught**: $500
- 📉 **Potential savings per 10K transactions**: $100K+
- 📈 **ROI uplift**: 8–15% depending on transaction volume and manual review reduction

---

## 💼 Strategic Recommendations

- Integrate model into **payment authorization flow**
- Flag transactions >90% fraud probability
- Retrain model **quarterly** with new data
- Combine SHAP + LIME for **compliance auditing**
- Enable manual review dashboards for borderline cases

---

## 🌐 Live App

Use the app here:                                                   
👉 (frauddetection-ml.streamlit.app)[https://frauddetection-analytics.streamlit.app/]

---

## 🔍 Project Structure

Fraud_Analytics/                                                 
├─ app.py                                                                         
├─ creditcard.csv (downloaded at runtime)         
├─ test_demo.csv (balanced demo sample for unbiased app testing)                                         
├─ model_xgb.pkl                                                           
├─ shap_explainer.pkl                                                                      
├─ lime_explanation_transaction_15.html                                   
├─ requirements.txt
├─ test_demo.py (script to generate test_demo.csv)                                                 
└─ Fraud_Identification_ML.ipynb                                                         

---

## ✅ How to Run Locally

git clone https://github.com/SweetySeelam2/Fraud_Detection_ML.git  
cd Fraud_Detection_ML  
pip install -r requirements.txt  
streamlit run app.py                                                                                          

---

## 👩‍💼 About the Author    

**Sweety Seelam** | Business Analyst | Aspiring Data Scientist | Passionate about building end-to-end ML solutions for real-world problems                                                                                                      
                                                                                                                                           
Email: sweetyseelam2@gmail.com                                                   

🔗 **Profile Links**                                                                                                                                                                       
[Portfolio Website](https://sweetyseelam2.github.io/SweetySeelam.github.io/)                                                         
[LinkedIn](https://www.linkedin.com/in/sweetyrao670/)                                                                   
[GitHub](https://github.com/SweetySeelam2)                                                             
[Medium](https://medium.com/@sweetyseelam)

---

## 🔐 Proprietary & All Rights Reserved
© 2025 Sweety Seelam. All rights reserved.

This project, including its source code, trained models, datasets (where applicable), visuals, and dashboard assets, is protected under copyright and made available for educational and demonstrative purposes only.

Unauthorized commercial use, redistribution, or duplication of any part of this project is strictly prohibited.
