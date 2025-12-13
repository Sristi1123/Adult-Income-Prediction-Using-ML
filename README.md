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
