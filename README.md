PayShield : Credit Card Delinquency Prediction (CDPM)

📌 Overview : 

This project focuses on predicting whether a credit card holder is likely to become delinquent (miss payments or default) based on their financial history and demographic details.
The goal is to assist financial institutions in identifying high-risk customers early, so they can take preventive measures like offering alternative repayment plans or adjusting credit limits.

This is a binary classification problem where the target variable indicates:

1 → Customer is likely to default or become delinquent

0 → Customer is not likely to become delinquent

📂 Dataset : 

Source: Provided by Forage as part of a Data Science & Analytics Virtual Experience Program.

Format: CSV file with both numeric and categorical features.

Target Column: Delinquency

📋 Features :

customer_id → Unique ID for each customer

age → Age of the customer (years)

gender → Gender (Male / Female)

income → Annual income in USD

credit_score → Credit rating score

loan_amount → Outstanding loan balance

loan_term → Duration of loan term (months or years)

previous_defaults → Number of previous loan defaults

employment_status → Employment type (Employed, Self-Employed, etc.)

Delinquency → Target variable (1 = Delinquent, 0 = Not Delinquent)


🛠 Project Workflow : 

🔍 Exploratory Data Analysis (EDA):

Examined data distribution, missing values, and outliers.

Checked feature correlations to understand relationships.

Visualized trends to spot patterns in delinquency behavior.


🤖 Predictive Modeling: 

Built multiple machine learning models (Random Forest, XGBoost, LightGBM, Logistic Regression).

Compared performance based on accuracy, recall, and speed.

Selected the most effective models for risk prediction.


🧪 Synthetic Data Generation : 

Used techniques like SMOTE (Synthetic Minority Oversampling Technique) to generate artificial samples for the minority class.

Why it’s useful:

Balances the dataset when real-world data is imbalanced.

Prevents the model from being biased toward the majority class.

Improves recall for rare but critical cases (e.g., customers likely to default).

Maintains data diversity without collecting more real data.


🛠️ Key Techniques Used :

Data Preprocessing:

Handling missing values (mean/median for numeric, mode for categorical)

Encoding categorical variables using Label Encoding / One-Hot Encoding

Scaling numeric features using StandardScaler for uniform range

Modeling :

🌲 Random Forest Classifier :

.Ensemble method combining many decision trees for strong predictions.
.Works well with imbalanced data, reduces overfitting, and handles mixed feature types.
.Result: ~99% accuracy, high recall — very effective at catching delinquent cases.

🚀 XGBoost:

.Boosting algorithm that builds trees sequentially to correct previous mistakes.
.Fast, memory-efficient, and well-suited for tabular data.
.Result: ~94% accuracy — slightly less than RF but faster in predictions.


💡 LightGBM:

.Leaf-wise boosting algorithm optimized for speed and large datasets.
.Low memory use, great for imbalanced data, and quick to train.
.Result: ~96–97% accuracy — very fast with solid performance.

📊 Logistic Regression :

.Simple linear model mapping features to probabilities.
.Easy to interpret but limited in capturing complex patterns.
.Result: ~55% accuracy — useful for comparison but underperformed on this dataset.


Model Evaluation:

Accuracy, Precision, Recall, F1-score

Confusion Matrix visualization for performance clarity

⚙️ Working Process: 

The workflow follows a structured machine learning pipeline:

1️⃣ Load & Inspect the Dataset 📂
Read the CSV dataset into Pandas DataFrame

Inspect shape, datatypes, missing values, and target distribution

2️⃣ Data Cleaning & Preprocessing 🧹
Handle missing values appropriately

Encode categorical variables into numerical format

Scale numeric features for model fairness

3️⃣ Train-Test Split ✂️
Split dataset into:

80% Training

20% Testing

Ensures evaluation on unseen data

4️⃣ Model Training 🤖
Random Forest Classifier — Ensemble-based, high accuracy

XGBoost Classifier — Boosting method, great recall and precision

Logistic Regression — Linear baseline model

5️⃣ Model Evaluation 📈
Test each model on the same dataset

Compare metrics to choose the best-performing algorithm

6️⃣ Predictions & Insights 💡
Best model predicts "At Risk" (1) or "Not At Risk" (0) customers

Identify top contributing features (e.g., credit score, previous defaults, income)

Recommendations for preventive actions by financial institutions


📊 Results :

Random Forest Classifier achieved 99% accuracy, making it the most reliable model for prediction.

XGBoost Classifier delivered 94% accuracy, performing strongly with balanced precision and recall.

Logistic Regression achieved only 55% accuracy, serving mainly as a baseline for comparison.


🏆 Conclusion:

Random Forest Classifier is the recommended model for deployment due to its exceptional accuracy.

XGBoost is a good backup option when faster inference or lighter computation is needed.

Logistic Regression underperforms for this dataset but is useful for explainability and as a baseline.

🚀 Future Improvements:

Try additional models like LightGBM, CatBoost, or Neural Networks

Implement cross-validation for more robust accuracy

Use feature selection to improve interpretability and reduce complexity

Deploy the model using Flask or Streamlit for interactive predictions

🖥️ Tech Stack :

Language: Python 🐍

Libraries: Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn

Tools: Jupyter Notebook, VS Code

Version Control: Git & GitHub

👨‍💻 Author:

Umashankar G
📧 Email: umashankargudivada@gmail.com
