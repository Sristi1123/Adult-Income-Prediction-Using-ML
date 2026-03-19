# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, roc_curve, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-left: 5px solid #2563EB;
        padding-left: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h3 {
        font-size: 1rem;
        margin: 0;
        opacity: 0.9;
    }
    .metric-card p {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1E3A8A;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .info-box {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = None

# Helper functions
@st.cache_data
def load_adult_dataset():
    """Load the Adult Income dataset"""
    try:
        # Try to load from local file
        if os.path.exists('UCI_Adult_Income_Dataset.csv'):
            df = pd.read_csv('UCI_Adult_Income_Dataset.csv')
            return df, "Adult Income Dataset"
        elif os.path.exists('adult.csv'):
            df = pd.read_csv('adult.csv')
            return df, "Adult Income Dataset"
        else:
            # Create synthetic Adult dataset
            return create_adult_dataset(), "Synthetic Adult Income Dataset"
    except Exception as e:
        st.error(f"Error loading Adult dataset: {str(e)}")
        return create_adult_dataset(), "Synthetic Adult Income Dataset (fallback)"

@st.cache_data
def create_adult_dataset():
    """Create a realistic sample of the Adult Income dataset"""
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 
                                       'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay'], 
                                      n_samples),
        'fnlwgt': np.random.randint(10000, 1500000, n_samples),
        'education': np.random.choice(['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc',
                                      '11th', 'Doctorate'], n_samples),
        'education-num': np.random.randint(1, 16, n_samples),
        'marital-status': np.random.choice(['Never-married', 'Married-civ-spouse', 'Divorced',
                                           'Separated', 'Widowed'], n_samples),
        'occupation': np.random.choice(['Prof-specialty', 'Craft-repair', 'Exec-managerial',
                                       'Adm-clerical', 'Sales', 'Other-service'], n_samples),
        'relationship': np.random.choice(['Husband', 'Not-in-family', 'Wife', 'Own-child'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo'], n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'capital-gain': np.random.choice([0, 0, 0, 1000, 5000, 10000], n_samples),
        'capital-loss': np.random.choice([0, 0, 0, 500, 1000, 2000], n_samples),
        'hours-per-week': np.random.randint(20, 80, n_samples),
        'native-country': np.random.choice(['United-States', 'Mexico', 'Canada', 'India'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic income
    income_score = (
        (df['age'] > 30) * 0.2 +
        (df['education-num'] > 12) * 0.3 +
        (df['hours-per-week'] > 40) * 0.2 +
        (df['capital-gain'] > 0) * 0.15
    )
    income_score = income_score / income_score.max()
    df['income'] = np.where(np.random.random(n_samples) < income_score, '>50K', '<=50K')
    
    return df

def is_adult_dataset(df):
    """Check if the dataset is the Adult Income dataset"""
    adult_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    # Check if most of the adult columns are present
    common_columns = set(df.columns) & set(adult_columns)
    return len(common_columns) >= 10  # If at least 10 adult columns match

# Header
st.markdown('<p class="main-header">📊 ML Classification Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>🌟 Welcome!</strong> This dashboard supports any classification dataset. 
    You can use the built-in Adult Income dataset or upload your own CSV file.
    Select your target column and choose models to train and compare.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/combo-chart.png", width=80)
    st.title("⚙️ Control Panel")
    
    # Data Source Selection
    st.subheader("📊 Data Source")
    data_option = st.radio(
        "Choose data source:",
        ["Use Adult Income Dataset", "Upload Your Own CSV"]
    )
    
    if data_option == "Upload Your Own CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.dataset_type = "custom"
                st.success(f"✅ Custom dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.session_state.data_loaded = False
        else:
            st.info("Please upload a CSV file")
            st.session_state.data_loaded = False
    else:
        # Load Adult dataset
        if st.session_state.df is None or st.session_state.dataset_type != "adult":
            with st.spinner("Loading Adult Income dataset..."):
                df, source = load_adult_dataset()
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.dataset_type = "adult"
                st.success(f"✅ {source}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        st.markdown("---")
        
        # Target Column Selection
        st.subheader("🎯 Target Column Selection")
        
        # Auto-detect target for Adult dataset
        if st.session_state.dataset_type == "adult" and 'income' in df.columns:
            default_target = 'income'
            st.info("✅ Auto-detected 'income' as target column for Adult dataset")
        else:
            default_target = df.columns[-1]  # Default to last column
        
        target_column = st.selectbox(
            "Select the target column:",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_target) if default_target in df.columns else 0
        )
        
        st.session_state.target_column = target_column
        
        # Dataset Overview
        st.markdown("---")
        st.subheader("📈 Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", f"{df.shape[0]:,}")
        with col2:
            st.metric("Features", df.shape[1])
        
        # Target distribution
        target_counts = df[target_column].value_counts()
        st.markdown("**Target Distribution:**")
        for label, count in target_counts.items():
            st.markdown(f"• {label}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Feature types
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        st.markdown(f"**Numerical features:** {len(numerical_cols)}")
        st.markdown(f"**Categorical features:** {len(categorical_cols)}")
        
        st.markdown("---")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        st.markdown("Select 4-5 models to train and compare:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_lr = st.checkbox("Logistic Regression", value=True)
            use_knn = st.checkbox("K-Nearest Neighbors", value=True)
            use_svm = st.checkbox("Support Vector Machine", value=False)
            use_dt = st.checkbox("Decision Tree", value=False)
        
        with col2:
            use_rf = st.checkbox("Random Forest", value=True)
            use_gb = st.checkbox("Gradient Boosting", value=False)
            use_xgb = st.checkbox("XGBoost", value=True)
            use_lgbm = st.checkbox("LightGBM", value=False)
        
        # Collect selected models
        selected_models = []
        if use_lr: selected_models.append("Logistic Regression")
        if use_knn: selected_models.append("KNN")
        if use_svm: selected_models.append("SVM")
        if use_dt: selected_models.append("Decision Tree")
        if use_rf: selected_models.append("Random Forest")
        if use_gb: selected_models.append("Gradient Boosting")
        if use_xgb: selected_models.append("XGBoost")
        if use_lgbm: selected_models.append("LightGBM")
        
        if len(selected_models) < 1:
            st.warning("⚠️ Please select at least one model")
        
        st.markdown(f"**Selected:** {len(selected_models)} models")
        
        # Training Options
        st.subheader("⚙️ Training Options")
        
        test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
        cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        random_state = st.number_input("Random seed", 0, 100, 42)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            handle_categorical = st.selectbox(
                "Handle categorical features:",
                ["One-Hot Encoding", "Label Encoding", "Drop"]
            )
            
            scaling_method = st.selectbox(
                "Feature scaling:",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"]
            )
            
            use_grid_search = st.checkbox("Use Grid Search (slower but better)", value=False)
            
            if use_grid_search:
                optimize_for = st.selectbox(
                    "Optimize for:",
                    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
                )
        
        # Training button
        train_button = st.button("🚀 Train Selected Models", use_container_width=True)

# Main content area
if st.session_state.data_loaded:
    df = st.session_state.df
    target_column = st.session_state.target_column
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Overview", "📊 Exploratory Analysis", 
        "🤖 Model Training", "📈 Results & Comparison"
    ])
    
    with tab1:
        st.markdown('<p class="sub-header">📋 Data Overview</p>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Samples</h3>
                <p>{df.shape[0]:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Features</h3>
                <p>{df.shape[1]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Missing Values</h3>
                <p>{missing:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            memory = df.memory_usage().sum() / 1024**2
            st.markdown(f"""
            <div class="metric-card">
                <h3>Memory Usage</h3>
                <p>{memory:.2f} MB</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Data Preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data Info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Numerical Features")
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if target_column in numerical_cols:
                numerical_cols.remove(target_column)
            st.write(numerical_cols if numerical_cols else "None")
            
            if numerical_cols:
                st.subheader("Numerical Statistics")
                st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        
        with col2:
            st.subheader("Categorical Features")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
            st.write(categorical_cols if categorical_cols else "None")
            
            if categorical_cols:
                st.subheader("Categorical Statistics")
                st.dataframe(df[categorical_cols].describe(), use_container_width=True)
    
    with tab2:
        st.markdown('<p class="sub-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
        
        # Target Distribution
        st.subheader("Target Variable Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            target_counts = df[target_column].value_counts().reset_index()
            target_counts.columns = ['Target', 'Count']
            fig = px.pie(target_counts, values='Count', names='Target', 
                        title=f'{target_column} Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart
            fig = px.bar(target_counts, x='Target', y='Count', 
                        title=f'{target_column} Counts',
                        color='Target',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Numerical feature distributions
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        if numerical_cols:
            st.subheader("Numerical Features Distribution")
            # Select which numerical feature to analyze
            selected_num = st.selectbox("Select numerical feature:", numerical_cols)
            
            fig = px.histogram(df, x=selected_num, color=target_column if target_column in df.columns else None,
                              marginal='box', barmode='overlay',
                              title=f'{selected_num} Distribution by {target_column}',
                              opacity=0.7)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Categorical feature distributions
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        if categorical_cols:
            st.subheader("Categorical Features Analysis")
            # Select which categorical feature to analyze
            selected_cat = st.selectbox("Select categorical feature:", categorical_cols)
            
            # Create cross-tabulation
            crosstab = pd.crosstab(df[selected_cat], df[target_column], normalize='index') * 100
            crosstab = crosstab.reset_index()
            crosstab_melted = crosstab.melt(id_vars=[selected_cat], 
                                           value_vars=df[target_column].unique(),
                                           var_name='Target', value_name='Percentage')
            
            fig = px.bar(crosstab_melted, x=selected_cat, y='Percentage', 
                       color='Target', title=f'{selected_cat} vs {target_column}',
                       barmode='group')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap for numerical features
        if len(numerical_cols) > 1:
            st.subheader("Correlation Heatmap")
            corr_matrix = df[numerical_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                          color_continuous_scale='RdBu_r',
                          title="Feature Correlations")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<p class="sub-header">🤖 Model Training</p>', unsafe_allow_html=True)
        
        if train_button and len(selected_models) >= 1:
            with st.spinner(f"Training {len(selected_models)} models... This may take a few minutes."):
                # Prepare data
                X = df.drop(target_column, axis=1)
                y = df[target_column]
                
                # Handle categorical features
                categorical_cols = X.select_dtypes(include=['object']).columns
                
                if len(categorical_cols) > 0:
                    if handle_categorical == "One-Hot Encoding":
                        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    elif handle_categorical == "Label Encoding":
                        for col in categorical_cols:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                    else:  # Drop
                        X = X.drop(columns=categorical_cols)
                
                # Encode target if it's categorical
                if y.dtype == 'object':
                    le_target = LabelEncoder()
                    y_encoded = le_target.fit_transform(y)
                    st.session_state.label_encoder = le_target
                    class_names = le_target.classes_
                else:
                    y_encoded = y.values
                    class_names = np.unique(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                )
                
                # Scale features
                if scaling_method != "None" and X_train.shape[1] > 0:
                    if scaling_method == "StandardScaler":
                        scaler = StandardScaler()
                    elif scaling_method == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()
                    
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    st.session_state.scaler = scaler
                else:
                    X_train_scaled = X_train.values if hasattr(X_train, 'values') else X_train
                    X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test
                
                # Define models with hyperparameters
                models = {
                    "Logistic Regression": {
                        'model': LogisticRegression(max_iter=1000, random_state=random_state),
                        'params': {'C': [0.1, 1, 10]}
                    },
                    "KNN": {
                        'model': KNeighborsClassifier(),
                        'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
                    },
                    "SVM": {
                        'model': SVC(probability=True, random_state=random_state),
                        'params': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
                    },
                    "Decision Tree": {
                        'model': DecisionTreeClassifier(random_state=random_state),
                        'params': {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10]}
                    },
                    "Random Forest": {
                        'model': RandomForestClassifier(random_state=random_state, n_jobs=-1),
                        'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
                    },
                    "Gradient Boosting": {
                        'model': GradientBoostingClassifier(random_state=random_state),
                        'params': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
                    },
                    "XGBoost": {
                        'model': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
                        'params': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
                    },
                    "LightGBM": {
                        'model': lgb.LGBMClassifier(random_state=random_state, verbose=-1),
                        'params': {'n_estimators': [100, 200], 'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1]}
                    }
                }
                
                # Train selected models
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, model_name in enumerate(selected_models):
                    if model_name in models:
                        status_text.text(f"Training {model_name}... ({idx + 1}/{len(selected_models)})")
                        
                        if use_grid_search and 'params' in models[model_name]:
                            # Determine scoring metric
                            scoring_map = {
                                "Accuracy": "accuracy",
                                "Precision": "precision_weighted",
                                "Recall": "recall_weighted",
                                "F1-Score": "f1_weighted",
                                "ROC-AUC": "roc_auc_ovr_weighted"
                            }
                            scoring = scoring_map.get(optimize_for, "accuracy")
                            
                            grid = GridSearchCV(
                                models[model_name]['model'],
                                models[model_name]['params'],
                                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                                scoring=scoring,
                                n_jobs=-1
                            )
                            grid.fit(X_train_scaled, y_train)
                            best_model = grid.best_estimator_
                            best_params = grid.best_params_
                        else:
                            best_model = models[model_name]['model']
                            best_model.fit(X_train_scaled, y_train)
                            best_params = "Default"
                        
                        # Predictions
                        y_pred = best_model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Handle binary vs multi-class
                        if len(class_names) == 2:
                            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                            precision = precision_score(y_test, y_pred)
                            recall = recall_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            roc_auc = roc_auc_score(y_test, y_pred_proba)
                        else:
                            y_pred_proba = best_model.predict_proba(X_test_scaled)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                        
                        # Cross-validation score
                        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                                   cv=cv_folds, scoring='accuracy')
                        
                        results.append({
                            'Model': model_name,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1,
                            'ROC-AUC': roc_auc if len(class_names) == 2 else 0,
                            'CV Mean': cv_scores.mean(),
                            'CV Std': cv_scores.std(),
                            'Best Params': str(best_params),
                            'Model Object': best_model
                        })
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(selected_models))
                
                status_text.text("Training completed!")
                time.sleep(1)
                
                # Store results
                results_df = pd.DataFrame(results)
                st.session_state.results = results_df
                st.session_state.X_test = X_test_scaled
                st.session_state.y_test = y_test
                st.session_state.class_names = class_names
                st.session_state.model_trained = True
                st.session_state.feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"Feature_{i}" for i in range(X.shape[1])]
                
                st.success(f"✅ Successfully trained {len(results)} models!")
                st.balloons()
                
                # Switch to results tab
                st.rerun()
        
        elif train_button and len(selected_models) < 1:
            st.warning("⚠️ Please select at least one model to train.")
    
    with tab4:
        st.markdown('<p class="sub-header">📈 Model Results & Comparison</p>', unsafe_allow_html=True)
        
        if st.session_state.model_trained:
            results_df = st.session_state.results
            
            # Sort by accuracy
            results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
            
            # Model ranking
            st.subheader("🏆 Model Performance Ranking")
            
            # Metrics comparison
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart comparison
                fig = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                if st.session_state.class_names is not None and len(st.session_state.class_names) == 2:
                    metrics.append('ROC-AUC')
                
                colors = px.colors.qualitative.Set3[:len(metrics)]
                
                for i, metric in enumerate(metrics):
                    if metric in results_df.columns:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=results_df['Model'],
                            y=results_df[metric],
                            text=results_df[metric].round(3),
                            textposition='auto',
                            marker_color=colors[i]
                        ))
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    barmode='group',
                    xaxis_title="Model",
                    yaxis_title="Score",
                    yaxis_range=[0, 1],
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Best model highlight
                best_model = results_df.iloc[0]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white;">
                    <h3 style="color: white; margin-bottom: 10px;">🏆 Best Model</h3>
                    <h2 style="font-size: 1.8rem; margin: 0; color: white;">{best_model['Model']}</h2>
                    <hr style="border-color: rgba(255,255,255,0.3);">
                    <p><strong>Accuracy:</strong> {best_model['Accuracy']:.2%}</p>
                    <p><strong>Precision:</strong> {best_model['Precision']:.2%}</p>
                    <p><strong>Recall:</strong> {best_model['Recall']:.2%}</p>
                    <p><strong>F1-Score:</strong> {best_model['F1-Score']:.2%}</p>
                    {f"<p><strong>ROC-AUC:</strong> {best_model['ROC-AUC']:.2%}</p>" if 'ROC-AUC' in best_model and len(st.session_state.class_names) == 2 else ""}
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed results table
            st.subheader("📊 Detailed Performance Metrics")
            display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            if 'ROC-AUC' in results_df.columns and len(st.session_state.class_names) == 2:
                display_cols.append('ROC-AUC')
            display_cols.extend(['CV Mean', 'CV Std'])
            
            display_df = results_df[display_cols].copy()
            for col in display_df.columns[1:]:
                display_df[col] = display_df[col].round(3)
            st.dataframe(display_df, use_container_width=True)
            
            # Confusion Matrices
            st.subheader("🔍 Confusion Matrices")
            
            # Create columns based on number of models
            num_models = len(results_df)
            cols_per_row = 3
            num_rows = (num_models + cols_per_row - 1) // cols_per_row
            
            for row in range(num_rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    model_idx = row * cols_per_row + col_idx
                    if model_idx < num_models:
                        with cols[col_idx]:
                            model = results_df.iloc[model_idx]['Model Object']
                            y_pred = model.predict(st.session_state.X_test)
                            cm = confusion_matrix(st.session_state.y_test, y_pred)
                            
                            fig = px.imshow(cm, text_auto=True, aspect='auto',
                                           color_continuous_scale='Blues',
                                           title=f"{results_df.iloc[model_idx]['Model']}",
                                           labels=dict(x="Predicted", y="Actual"))
                            fig.update_layout(height=300, coloraxis_showscale=False)
                            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            tree_models = ['Random Forest', 'XGBoost', 'LightGBM', 'Decision Tree', 'Gradient Boosting']
            available_tree_models = [m for m in tree_models if m in results_df['Model'].values]
            
            if available_tree_models and hasattr(results_df.iloc[0]['Model Object'], 'feature_importances_'):
                st.subheader("🔍 Feature Importance Analysis")
                selected_tree = st.selectbox("Select model for feature importance:", available_tree_models)
                model_obj = results_df[results_df['Model'] == selected_tree]['Model Object'].iloc[0]
                
                if hasattr(model_obj, 'feature_importances_'):
                    importances = model_obj.feature_importances_
                    feature_imp = pd.DataFrame({
                        'Feature': st.session_state.feature_names[:len(importances)],
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                               title=f"Top 15 Features - {selected_tree}",
                               color='Importance', color_continuous_scale='Viridis')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curves (only for binary classification)
            if len(st.session_state.class_names) == 2:
                st.subheader("📈 ROC Curves Comparison")
                fig = go.Figure()
                
                for _, row in results_df.iterrows():
                    model = row['Model Object']
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
                        
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            name=f"{row['Model']} (AUC={row['ROC-AUC']:.3f})",
                            mode='lines',
                            line=dict(width=2)
                        ))
                
                # Add diagonal line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random (AUC=0.500)',
                    mode='lines',
                    line=dict(dash='dash', color='gray', width=1)
                ))
                
                fig.update_layout(
                    title="ROC Curves",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=500,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👈 Please train models in the 'Model Training' tab first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>ML Classification Dashboard | Built with Streamlit & Scikit-learn</p>
    <p style="font-size: 0.8rem;">Supports any classification dataset | Choose your target column and compare multiple models</p>
</div>
""", unsafe_allow_html=True)