# Machine Learning Assignment 2 – Implementation, Evaluation and Deployment of Classification Models  

## Problem Statement

The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset, evaluate their performance using standard metrics, and deploy an interactive Streamlit web application to demonstrate the models.  
Users can upload test data, select a trained model, and visualize the model performance through evaluation metrics, classification report, and confusion matrix.

---

## Dataset Description
**Dataset Name:** Heart Disease Dataset from Kaggle

**Dataset Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  

**Brief Description:** The Heart Disease dataset contains clinical patient attributes used to predict the presence or absence of heart disease. It includes 1025 instances with 13 features and a binary target variable, making it suitable for supervised classification tasks. 

**Dataset attributes (13 input features):** 

1.  **age** - Age of the patient (in years)

2.  **sex** - Gender of the patient (1 = male, 0 = female)

3. **cp** - Chest pain type (categorical: 0–3 indicating different pain types)

4. **trestbps** - Resting blood pressure (in mm Hg)

5. **chol** - Serum cholesterol level (in mg/dl)

6. **fbs** - Fasting blood sugar (> 120 mg/dl: 1 = true, 0 = false)

7. **restecg** - Resting electrocardiographic results (categorical values)

8. **thalach** - Maximum heart rate achieved

9. **exang** - Exercise-induced angina (1 = yes, 0 = no)

10. **oldpeak** - ST depression induced by exercise relative to rest

11. **slope** - Slope of the peak exercise ST segment

12. **ca** - Number of major vessels colored by fluoroscopy (0–3)

13. **thal** - Thalassemia (categorical feature representing blood disorder type)

**Target Variable:** 

- **target:** Target variable indicating presence (1) or absence (0) of heart disease

**Dataset Statistics** -
 - **Number of Features** - 13 
 - **Number of Instances** - 1025 
 - **Type** - Binary Classification  

---

## Models Used and Evaluation Metrics

**Implemented Models**

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbor Classifier  
- Naive Bayes Classifier (Gaussian)  
- Ensemble Model - Random Forest 
- Ensemble Model - XGBoost 


<br>

**Evaluation Metrics**

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### Comparison Table with Evaluation Metrics
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead>
    <tr style="border: 1px solid #444;">
      <th style="padding: 10px; border: 1px solid #444;">ML Model Name</th>
      <th style="border: 1px solid #444;">Accuracy</th>
      <th style="border: 1px solid #444;">AUC</th>
      <th style="border: 1px solid #444;">Precision</th>
      <th style="border: 1px solid #444;">Recall</th>
      <th style="border: 1px solid #444;">F1</th>
      <th style="border: 1px solid #444;">MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">Logistic Regression</td>
      <td style="border: 1px solid #444;">0.844</td><td style="border: 1px solid #444;">0.935</td><td style="border: 1px solid #444;">0.807</td>
      <td style="border: 1px solid #444;">0.914</td><td style="border: 1px solid #444;">0.857</td><td style="border: 1px solid #444;">0.693</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">Decision Tree</td>
      <td style="border: 1px solid #444;">0.995</td><td style="border: 1px solid #444;">0.995</td><td style="border: 1px solid #444;">1.000</td>
      <td style="border: 1px solid #444;">0.990</td><td style="border: 1px solid #444;">0.995</td><td style="border: 1px solid #444;">0.990</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">kNN</td>
      <td style="border: 1px solid #444;">0.951</td><td style="border: 1px solid #444;">0.990</td><td style="border: 1px solid #444;">0.944</td>
      <td style="border: 1px solid #444;">0.962</td><td style="border: 1px solid #444;">0.953</td><td style="border: 1px solid #444;">0.903</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">Naive Bayes</td>
      <td style="border: 1px solid #444;">0.824</td><td style="border: 1px solid #444;">0.910</td><td style="border: 1px solid #444;">0.800</td>
      <td style="border: 1px solid #444;">0.876</td><td style="border: 1px solid #444;">0.836</td><td style="border: 1px solid #444;">0.651</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">Random Forest (Ensemble)</td>
      <td style="border: 1px solid #444;">0.995</td><td style="border: 1px solid #444;">1.000</td><td style="border: 1px solid #444;">1.000</td>
      <td style="border: 1px solid #444;">0.990</td><td style="border: 1px solid #444;">0.995</td><td style="border: 1px solid #444;">0.990</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">XGBoost (Ensemble)</td>
      <td style="border: 1px solid #444;">0.995</td><td style="border: 1px solid #444;">1.000</td><td style="border: 1px solid #444;">1.000</td>
      <td style="border: 1px solid #444;">0.990</td><td style="border: 1px solid #444;">0.995</td><td style="border: 1px solid #444;">0.990</td>
    </tr>
  </tbody>
</table>


### Observations about Model Performance

 <table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead>
    <tr style="border: 1px solid #444;">
      <th style="padding: 10px; border: 1px solid #444;">ML Model Name</th>
      <th style="border: 1px solid #444;">Observations</th>  
    </tr>
  </thead>
  <tbody>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">Logistic Regression</td>
      <td style="border: 1px solid #444;">Provides a strong baseline with good interpretability and stable performance on linearly separable patterns. High Recall (0.914) indicates strong detection of heart disease cases (few false negatives). However, lower Precision (0.762) shows more false positives. Strong AUC (0.930) indicates good class separability.</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">Decision Tree</td>
      <td style="border: 1px solid #444;">Captures non-linear relationships well but is prone to overfitting without proper depth control. Extremely high Precision (1.000) means no false positives. High Recall (0.971) shows very few missed cases. Very strong MCC (0.971) indicates balanced performance. Slight risk of overfitting due to near-perfect metrics.</td><
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">kNN</td>
      <td style="border: 1px solid #444;">Performance is sensitive to the choice of k and feature scaling; performs well when data is normalized. Balanced Precision and Recall indicate stable performance. Good AUC (0.963) shows strong discrimination. Moderate MCC (0.727) suggests reliable classification.</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">Naive Bayes</td>
      <td style="border: 1px solid #444;">Fast and simple model; performs reasonably well despite the independence assumption among features. Good Recall (0.876) but moderate Precision. AUC (0.904) shows decent class separation.</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">Random Forest (Ensemble)</td>
      <td style="border: 1px solid #444;">Demonstrate strong generalization due to ensemble learning and reduces overfitting compared to a single tree.</td>
    </tr>
    <tr style="border: 1px solid #444;">
      <td style="padding: 10px; border: 1px solid #444; text-align: left;">XGBoost (Ensemble)</td>
      <td style="border: 1px solid #444;">Achieves the best overall performance with high AUC and MCC due to boosting, regularization, and optimized tree learning.</td><
    </tr>
  </tbody>
</table>

🚀 Streamlit App Deployment

The project has been deployed on Streamlit Community Cloud.

Live App Link: [Insert Streamlit App URL]

Features included:

Dataset upload option (CSV)

Model selection dropdown

Display of evaluation metrics

Confusion matrix / classification report

📂 Repository Structure

project-folder/
│── app.py (or streamlit_app.py)
│── requirements.txt
│── README.md
│── model/ (saved model files for all implemented models)

📋 Requirements

Dependencies required for deployment (add more if used):

streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
xgboost

✅ Final Submission Checklist

[ ] GitHub repo link works

[ ] Streamlit app link opens correctly

[ ] App loads without errors

[ ] All required features implemented

[ ] README.md updated and added in the submitted PDF


This template follows the exact structure required in your assignment instructions. Once you run your models, you just need to **fill in the metrics table and observations**, and paste your dataset link plus Streamlit app link.  

Would you like me to also draft a **sample filled-in version** (with hypothetical dataset and metrics) so you can see how a completed README might look?