# Customer-Churn-Analysis-Prediction---Python
# 📊 Customer Churn Analysis & Prediction

### Python | Machine Learning | Databricks | Customer Analytics

## 1. Project Overview

This project looks at customer churn and how machine learning can help a business identify customers who may be at risk of leaving.

Customer churn happens when customers stop using a company's products or services. Losing customers can affect business revenue, growth and long-term customer relationships.

The aim of this project was to explore customer information, identify patterns linked to customer churn, and build a machine learning model that can predict whether a customer is likely to leave.

I used the Telco Customer Churn dataset and Python in Databricks to complete the analysis and develop a Random Forest classification model.

## 2. Business Problem

For a business, gaining new customers is important, but keeping existing customers is just as important.

When customers leave, the business loses potential future revenue and may need to spend more money attracting new customers.

This analysis focuses on the following questions:

- How many customers are leaving the company?
- What customer information can help us understand churn?
- Can we use machine learning to predict which customers are likely to leave?
- How can the business use these predictions to improve customer retention?

## 3. Dataset Description

The project uses the Telco Customer Churn dataset, which contains customer information, service details, payment methods and churn status.

Some of the columns explored include:

| Column | Description |
|---|---|
| `customerID` | Unique customer reference |
| `gender` | Customer gender |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed |
| `InternetService` | Type of internet service |
| `Contract` | Customer contract type |
| `PaymentMethod` | How the customer pays |
| `MonthlyCharges` | Monthly amount charged |
| `TotalCharges` | Total amount charged to the customer |
| `Churn` | Whether the customer left the company |

The `Churn` column is the target variable used to train the prediction model.

- Yes – The customer left.
- No – The customer stayed.

## 4. Tools and Technologies

The following tools and Python libraries were used:

- **Databricks:** To run the notebook and access the dataset.
- **Python:** Main programming language.
- **Pandas:** To load, explore and prepare the data.
- **NumPy:** For numerical operations.
- **Seaborn:** To create visualisations.
- **Matplotlib:** To display charts.
- **Scikit-learn:** To prepare the data, train the model and evaluate its performance.

## 5. Project Workflow

I followed these steps to complete the analysis:

### Step 1: Loading and Understanding the Data

I loaded the Telco Customer Churn dataset from a Databricks table and explored the first few records.

I also checked for missing values and reviewed descriptive statistics to understand the information available.

### Step 2: Exploring Customer Churn

I used a count plot to compare customers who stayed with the company against those who left.

This provided an initial understanding of the churn distribution.

### Step 3: Data Preprocessing

Before training the model, I prepared the dataset by:

- Converting `TotalCharges` into a numerical column.
- Handling missing values using the median.
- Converting categorical columns into numerical values using Label Encoding.
- Removing `customerID`, as it is an identifier and does not provide useful predictive information.

### Step 4: Splitting the Dataset

I separated the customer information (features) from the churn outcome (target).

The dataset was divided into:

- 80% training data.
- 20% testing data.

This allowed me to train the model on one portion of the data and evaluate it on customer records it had not seen during training.

### Step 5: Feature Scaling

I used StandardScaler to bring numerical features onto a similar scale.

The scaler was fitted on the training data and then applied to the testing data.

### Step 6: Building the Machine Learning Model

I trained a Random Forest Classifier.

This algorithm uses multiple decision trees and combines their predictions to make a final classification.

The model was used to predict whether customers in the test dataset would stay or leave.

### Step 7: Evaluating the Model

I compared the actual customer outcomes against the model's predictions.

I used the following evaluation methods:

- Accuracy score.
- Confusion matrix.
- Correctly predicted customers.
- Incorrect predictions.
- Customers who left but were missed by the model.

## 6. Model Evaluation and Results

The Random Forest model was evaluated using the test dataset.

The confusion matrix showed the following results:

| Prediction outcome | Number of customers |
|---|---:|
| Correctly predicted customers who stayed | 924 |
| Correctly predicted customers who left | 181 |
| Customers predicted to leave but who stayed | 117 |
| Customers who left but were predicted to stay | 187 |

### What do these results tell us?

The model correctly identified 924 customers who stayed and 181 customers who left.

However, it also incorrectly classified 117 customers as potential churners and failed to identify 187 customers who actually left.

The missed churn cases are particularly important from a business perspective. These are customers the company could have targeted with retention campaigns but the model failed to identify.

This suggests that the model could benefit from further improvement, especially in identifying customers who are likely to leave.

## 7. Business Recommendations

Based on the purpose of this analysis, the business could consider the following actions:

1. **Improve customer retention:** Identify customers who may be at risk of leaving and contact them before they cancel their services.
2. **Review customer contracts:** Investigate whether customers on certain contract types are more likely to leave.
3. **Monitor payment behaviour:** Explore whether payment methods and monthly charges are associated with churn.
4. **Improve prediction accuracy:** Further develop the model to reduce the number of customers who leave without being identified as potential churners.
5. **Use data for decision-making:** Include churn predictions in customer retention reporting to help management make informed decisions.

## 8. Areas for Improvement

There are several ways this project could be developed further:

- Compare Random Forest with Logistic Regression and Decision Tree models.
- Tune model parameters to improve performance.
- Evaluate precision, recall and F1-score.
- Investigate which customer features have the greatest influence on churn.
- Address class imbalance where necessary.
- Explore methods of improving the model's ability to identify customers who leave.

## 9. Key Learning Outcomes

Through this project, I gained practical experience in:

- Working with customer datasets in Databricks.
- Exploring and understanding data using Python.
- Cleaning missing and incorrect values.
- Preparing categorical data for machine learning.
- Splitting data into training and testing sets.
- Applying feature scaling.
- Building a Random Forest classification model.
- Evaluating predictions using a confusion matrix.
- Connecting machine learning results to business decisions.

## 10. Conclusion

This project helped me understand how customer data can be used to address a real business problem.

Building the Random Forest model was an important step in learning how machine learning can support customer retention.

One of the main lessons from this exercise is that accuracy alone does not tell the whole story. A model also needs to identify customers who are likely to leave, because missing these customers can have a direct impact on business revenue.

Customer churn prediction is therefore not just about building a model. It is about using the results to help a business understand its customers, reduce customer losses and make better decisions.

---

**Project Category:** Data Analytics | Machine Learning | Classification

**Environment:** Databricks Notebook

**Language:** Python

**Dataset:** Telco Customer Churn
