# Adult-Income-Prediction-Using-ML
This project predicts whether a person’s income is greater than or less than $50K per year using Machine Learning models.
The dataset used is the Adult Income (Census Income Dataset).

## 🚀 Project Setup
- Loaded UCI Adult Income Dataset (48,842 records, 15 features)
- Initial data exploration:
  - Dataset shape and structure
  - Basic statistics for numerical features
  - Check for missing values
- Identified features like age, education, occupation, income category

## 🔧 Data Preprocessing
- Handled missing values:
  - Categorical: Filled with mode
  - Numerical: Filled with median
- Addressed outliers using IQR method
- Removed extreme values from features:
  - age: 216 outliers
  - fnlwgt: 1,453 outliers
  - hours-per-week: 13,496 outliers
- Generated correlation heatmap to understand feature relationships

## 🤖 Model Implementation
Implemented three classification models:

### Models Used:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**

### Preprocessing Steps:
- Label encoding for categorical variables
- Standard scaling for numerical features
- Train-test split (80-20 ratio)

### Evaluation Metrics Tracked:
- Accuracy
- Precision
- Recall
- Confusion Matrix


### Visualizations Included:
1. Confusion matrices for all models (subplot)
2. Clustered bar chart comparing metrics
3. Correlation heatmap of numerical features

## 🛠 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
