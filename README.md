# Overview
This project aims to detect fraudulent credit card transactions using unsupervised learning techniques. Fraud detection is critical for financial institutions to minimize losses and maintain trust. By analyzing transaction data, we can identify patterns that indicate potential fraud and take action before any damage occurs.

## Table of Contents
Project Motivation
Data Description
Approach
Modeling
Evaluation
Results
Future Work
Getting Started
Dependencies
Contributing
License

### Project Motivation
Credit card fraud detection is a challenging problem due to the imbalance in the dataset—fraudulent transactions are rare compared to legitimate ones. Traditional supervised learning approaches may struggle with this imbalance, making unsupervised learning an effective alternative. This project leverages techniques such as clustering, anomaly detection, and dimensionality reduction to identify potentially fraudulent transactions.

### Data Description
The dataset used in this project consists of anonymized credit card transactions. It includes the following features:

V1-V28: Principal components obtained using PCA.
Time: Seconds elapsed between this transaction and the first transaction in the dataset.
Amount: Transaction amount.
Class: Label indicating whether the transaction is fraudulent (1) or not (0).
The data is highly imbalanced, with fraudulent transactions accounting for a small fraction of the total transactions.

### Approach
The approach is inspired by Chapter 4 of "Hands-On Unsupervised Learning Using Python," which emphasizes the following steps:

Data Preprocessing: Handle missing values, normalize the features, and perform dimensionality reduction using PCA.
Clustering: Apply clustering algorithms like K-Means and DBSCAN to group similar transactions and identify anomalies.
Anomaly Detection: Use techniques like Isolation Forest and Local Outlier Factor (LOF) to detect outliers that may correspond to fraudulent transactions.
Evaluation: Assess the performance of the unsupervised models using metrics such as precision, recall, and the Area Under the Precision-Recall Curve (AUPRC).
### Modeling
Several unsupervised learning models were employed:

Principal Component Analysis (PCA): Reduced dimensionality to focus on the most important features.
K-Means Clustering: Grouped transactions into clusters to identify potential fraud.
DBSCAN: Detected dense regions in the data, treating sparsely populated areas as anomalies.
Isolation Forest: Identified anomalies by isolating points in a tree structure.
Local Outlier Factor (LOF): Evaluated the local density deviation of a data point relative to its neighbors.

### Evaluation
Given the unsupervised nature of the models, evaluation was primarily based on the ability to correctly identify fraudulent transactions (True Positives) while minimizing false positives. Metrics such as precision, recall, and F1-score were used, along with visualizations of clusters and outliers.

### Results
The models successfully identified a significant portion of fraudulent transactions, with varying levels of precision and recall. The use of unsupervised learning allowed for the detection of anomalies in a dataset where fraudulent transactions were not explicitly labeled during training.

### Future Work
Model Tuning: Further optimize hyperparameters for better detection accuracy.
Ensemble Methods: Combine multiple unsupervised models to improve detection rates.
Feature Engineering: Explore additional feature creation techniques to enhance model performance.
Supervised Learning Integration: Compare results with supervised models to assess the effectiveness of unsupervised learning.

### Getting Started
To reproduce this project, follow these steps:

Clone the repository:

git clone https://github.com/riasterdom/Credit-Card-Fraud-Detection.git
Navigate to the project directory:

cd Credit-Card-Fraud-Detection

Run the Jupyter notebooks:
jupyter notebook

### Dependencies
* Python 3.6
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
