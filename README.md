

## Introduction
Credit card fraud is a significant concern for financial institutions and consumers alike.
Detecting fraudulent transactions in real-time is crucial to minimize losses and maintain customer trust.
In this project, we developed a machine learning model to predict fraudulent credit card transactions using a dataset of anonymized credit card transactions.
The goal was to build a robust model that can accurately classify transactions as fraudulent or legitimate.

## Dataset Overview
The dataset used in this project contains credit card transactions made by European cardholders in September 2013.
It includes:

284,807 transactions, of which 492 are fraudulent (highly imbalanced dataset).

30 features: Time, Amount, and 28 anonymized features (V1 to V28) resulting from a PCA transformation for privacy protection.

Target variable: Class (1 for fraudulent, 0 for legitimate).

The dataset is highly imbalanced, with only 0.172% of transactions being fraudulent, making it a challenging classification problem.

## Business Problem
Credit card fraud detection is a critical business problem for financial institutions.
### The primary challenges are:

Imbalanced Data: Fraudulent transactions are rare compared to legitimate ones.

Real-Time Detection: Fraud detection systems must operate in real-time to prevent losses.

High Precision: False positives (legitimate transactions flagged as fraud) can lead to customer dissatisfaction, while false negatives (fraudulent transactions missed) result in financial losses.

The goal of this project was to build a machine learning model that can accurately detect fraudulent transactions while minimizing false positives and false negatives.

## Step-by-Step Approach

###Data Exploration:

Analyzed the dataset to understand its structure, distribution, and imbalance.

Visualized feature distributions and correlations.

### Data Preprocessing:

Scaled the Time and Amount features using StandardScaler.

Addressed the class imbalance using techniques like SMOTE (Synthetic Minority Oversampling Technique).

### Feature Selection:

Used SelectKBest to select the top 10 most important features for model training.

Reduced dimensionality to improve model performance and reduce overfitting.

### Model Training:

Trained multiple machine learning models, including:

Random Forest

XGBoost

Logistic Regression

Evaluated models using metrics like precision, recall, F1-score, and AUC-ROC.

### Model Evaluation:

Compared the performance of the models using a test set.

Selected the best-performing model based on AUC-ROC and precision-recall trade-off.

Visualization:

Plotted ROC curves and confusion matrices to visualize model performance.

## Challenges & Solutions
### Imbalanced Dataset:

Challenge: The dataset was highly imbalanced, with only 0.172% fraudulent transactions.

Solution: Used SMOTE to oversample the minority class and improve model performance.

### Feature Selection:

Challenge: The dataset had 30 features, some of which were irrelevant or redundant.

Solution: Used SelectKBest to select the top 10 most important features, reducing dimensionality and improving model efficiency.

### Model Overfitting:

Challenge: Some models (e.g., Random Forest) showed signs of overfitting.

Solution: Tuned hyperparameters using GridSearchCV and cross-validation to improve generalization.

### Feature Name Mismatch:

Challenge: The input data for predictions did not match the feature names used during training.

Solution: Ensured that the input data included all features used during training, even if some were not selected by the feature selector.

## Results
Best Model: Random Forest achieved the highest AUC-ROC score (0.98) and demonstrated a good balance between precision and recall.

Key Metrics:

Precision: 0.92

Recall: 0.85

F1-Score: 0.88

AUC-ROC: 0.98

The model successfully identified 85% of fraudulent transactions while maintaining a low false positive rate.

## Future Work
Real-Time Deployment:

Deploy the model as a real-time fraud detection system using Streamlit or Flask.

Advanced Models:

Experiment with deep learning models (e.g., LSTM) for sequence-based fraud detection.

Feature Engineering:

Explore additional feature engineering techniques to improve model performance.

Explainability:

Use SHAP or LIME to explain model predictions and improve transparency.

## Installation & How to Run
Clone the Repository:

bash
Copy
git clone https://github.com/your-username/credit-card-fraud-detection.git<br>
cd credit-card-fraud-detection<br>
Install Dependencies:

bash
Copy
pip install -r requirements.txt<br>
Run the Jupyter Notebook:

bash
Copy
jupyter notebook Credit_Card_Fraud_Detection.ipynb<br>
Run the Streamlit App (for future deployment):

bash
Copy
streamlit run app.py<br>
## Key Takeaways
Imbalanced Data Handling: Successfully addressed class imbalance using SMOTE, improving model performance on the minority class.

Feature Selection: Reduced dimensionality using SelectKBest, improving model efficiency and reducing overfitting.

Model Evaluation: Demonstrated a strong understanding of evaluation metrics (precision, recall, F1-score, AUC-ROC) and their trade-offs.

Problem-Solving: Overcame challenges like feature name mismatches and model overfitting through systematic debugging and tuning.

## Why is this proect Important
Highlight Technical Skills: This project demonstrates expertise in data preprocessing, feature selection, model training, and evaluation.

Showcase Problem-Solving: The ability to address challenges like imbalanced data and feature name mismatches highlights strong problem-solving skills.

Emphasize Real-World Impact: Fraud detection is a critical real-world problem, and this project showcases the ability to build practical, impactful solutions.

Future Potential: The project is scalable and can be extended to real-time deployment or advanced models, demonstrating forward-thinking.
