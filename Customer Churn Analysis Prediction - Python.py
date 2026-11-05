# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "seaborn",
# ]
# ///
# MAGIC %md
# MAGIC ## Customer Churn Analysis Prediction - Python

# COMMAND ----------

# MAGIC %md
# MAGIC Customer churn occurs when a customer stops using a company’s service lead to revenue loss. Analyzing churn helps businesses understand why customers leave and how to improve retention. High churn rates can affect revenue and business growth. By analyzing churn patterns businesses can take proactive steps to retain customers.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Importing Libraries and Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC Loading the Dataset
# MAGIC We start by importing the necessary Python libraries and loading the Telco Customer Churn dataset. This dataset contains various customer details such as service plans, usage behavior and churn status

# COMMAND ----------

# MAGIC %pip install seaborn 

# COMMAND ----------

# DBTITLE 1,Loading the Dataset
import numpy as np
import pandas as pd

# load the dataset from the Unity Catalog table
dataset = spark.read.table('workspace.default.telco_customer_churn').toPandas()

dataset.head()

# COMMAND ----------

# DBTITLE 1,Understanding the Dataset
print(dataset.isnull().sum())


# COMMAND ----------

print(dataset.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC **tenure** – The number of months a customer has stayed with the company.
# MAGIC **InternetServic**e – The type of internet service the customer has DSL, Fiber optic or None.
# MAGIC **PaymentMethod**– The method the customer uses for payments.
# MAGIC **Churn** – The target variable i.e Yes for customer churned and No for customer stayed.

# COMMAND ----------

# DBTITLE 1,Analyzing Churn Distribution
import seaborn as sns
import matplotlib.pyplot as plt

print(dataset['Churn'].value_counts())
sns.countplot(x='Churn', data=dataset, palette='coolwarm')
plt.title('Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **2. Data Preprocessing**

# COMMAND ----------

# MAGIC %md
# MAGIC **Handling Missing and Incorrect Values**
# MAGIC Before processing we ensure that all numerical columns contain valid values. The TotalCharges column sometimes has empty spaces which need to be converted to numerical values.
# MAGIC
# MAGIC pd.to_numeric(dataset['TotalCharges'], errors='coerce') converts the TotalCharges column to numerical format. If any value is not convertible (e.g., empty spaces), it replaces it with NaN.
# MAGIC .fillna(dataset['TotalCharges'].median(), inplace=True) replaces missing values (NaN) with the median of the column to maintain consistency in numerical values.

# COMMAND ----------

dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')


# COMMAND ----------

dataset['TotalCharges'] = dataset['TotalCharges'].fillna(dataset['TotalCharges'].median())

# COMMAND ----------

# MAGIC %md
# MAGIC **Handling Categorical Variables**
# MAGIC Some features like State, International Plan and Voice Mail Plan are **categorical **and must be converted into numerical values for model training.
# MAGIC
# MAGIC **LabelEncoder()** converts categorical values into numerical form. Each unique category is assigned a numeric label.
# MAGIC The loop iterates through each categorical column and applies fit_transform() to encode categorical variables into numbers.

# COMMAND ----------

# DBTITLE 1,Handling Categorical Variables
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
for col in categorical_cols:
    dataset[col] = labelencoder.fit_transform(dataset[col])

# COMMAND ----------

# MAGIC %md
# MAGIC **Feature Selection and Splitting Data**

# COMMAND ----------

# MAGIC %md
# MAGIC We separate the features (X) and target variable (y) and split the dataset into **training and testing sets.**
# MAGIC
# MAGIC X = dataset.drop(['customerID', 'Churn'], axis=1) removes the customerID (irrelevant for prediction) and Churn column (target variable).
# MAGIC
# MAGIC y = dataset['Churn'] defines y as the target variable, which we want to predict.
# MAGIC train_test_split() splits data into 80% training and 20% testing for model evaluation.

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = dataset.drop(['customerID', 'Churn'], axis=1)
y = dataset['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# COMMAND ----------

# MAGIC %md
# MAGIC **Feature Scaling**

# COMMAND ----------

# MAGIC %md
# MAGIC Since features are on different scales we apply **standardization** to improve model performance. It prevents models from being biased toward larger numerical values and improves convergence speed in optimization algorithms like gradient descent
# MAGIC
# MAGIC **StandardScaler()**: Standardizes data by transforming it to have a mean of 0 and a standard deviation of 1 ensuring all features are on a similar scale.
# MAGIC **fit_transform(X_train)**: Fits the scaler to the training data and transforms it.
# MAGIC **transform(X_test)**: Transforms the test data using the same scaling parameters.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC **3. Model Training and Prediction**

# COMMAND ----------

# MAGIC %md
# MAGIC For training our model we use **Random Forest Classifier**. It is an ensemble learning method that combines the results of multiple decision trees to make a final prediction

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# Print the predicted classes
print(y_pred)
print(clf)

# COMMAND ----------

# show the first 10 predictions next to the actual churn values
predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
display(predictions.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **4. Model Evaluation**

# COMMAND ----------

# MAGIC %md
# MAGIC **Accuracy Score**-To measure model performance we calculate accuracy using the accuracy_score function.

# COMMAND ----------

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Confusion Matrix and Performance Metrics**

# COMMAND ----------

# MAGIC %md
# MAGIC We evaluate precision, recall and accuracy using a confusion matrix.

# COMMAND ----------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="coolwarm")
plt.title('Confusion Matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Confusion matrix** shows how well the model predicts customer churn. It correctly identifies 924 non-churners and 181 churners. However 117 non-churners are wrongly classified as churners and 187 churners are missed. The high number of missed churners suggests the model may need further tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC