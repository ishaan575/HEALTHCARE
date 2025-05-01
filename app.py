import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Import custom modules
import data_loader
import data_exploration
import data_preprocessing
import feature_engineering
import model_training
import model_evaluation
import utils

# Set page configuration
st.set_page_config(
    page_title="Healthcare ML Tutorial",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'features_selected' not in st.session_state:
    st.session_state.features_selected = None

# Define the steps in the ML workflow
steps = [
    "Introduction",
    "Problem Definition",
    "Data Collection",
    "Data Exploration (EDA)",
    "Data Preprocessing",
    "Feature Engineering",
    "Model Selection & Training",
    "Model Evaluation",
    "Conclusion"
]

# Sidebar for navigation
st.sidebar.title("Healthcare ML Tutorial")
st.sidebar.write("A step-by-step guide to machine learning")

# Allow users to navigate through steps
selected_step = st.sidebar.radio("Navigate to step:", steps, index=st.session_state.current_step)
st.session_state.current_step = steps.index(selected_step)

# Add a note in the sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    "This application guides beginners through the complete machine learning workflow "
    "using a healthcare dataset for heart disease prediction."
)

# Main content
def main():
    if selected_step == "Introduction":
        show_introduction()
    elif selected_step == "Problem Definition":
        show_problem_definition()
    elif selected_step == "Data Collection":
        show_data_collection()
    elif selected_step == "Data Exploration (EDA)":
        show_data_exploration()
    elif selected_step == "Data Preprocessing":
        show_data_preprocessing()
    elif selected_step == "Feature Engineering":
        show_feature_engineering()
    elif selected_step == "Model Selection & Training":
        show_model_training()
    elif selected_step == "Model Evaluation":
        show_model_evaluation()
    elif selected_step == "Conclusion":
        show_conclusion()

def show_introduction():
    st.title("Introduction to Machine Learning in Healthcare")
    st.write("""
    ## Welcome to the Healthcare ML Tutorial!
    
    This application will guide you through the complete machine learning workflow from start to finish. 
    You'll learn how to use machine learning to predict heart disease based on patient data.
    
    ### What You'll Learn:
    - How to define a healthcare ML problem
    - Methods for exploring and understanding healthcare data
    - Techniques for preprocessing and preparing data
    - Feature engineering to improve model performance
    - Selecting and training appropriate ML models
    - Evaluating model performance with healthcare-specific metrics
    
    ### Who This Is For:
    This tutorial is designed for beginners in machine learning, especially those interested in healthcare applications.
    You don't need extensive programming or ML knowledge to follow along!
    
    ### Navigation:
    Use the sidebar to navigate through each step of the ML workflow. We recommend following the steps in order.
    
    Let's get started on your machine learning journey in healthcare!
    """)
    
    st.image("https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg", width=100)
    
    # Button to move to the next step
    if st.button("Next: Problem Definition →"):
        st.session_state.current_step = 1
        st.rerun()

def show_problem_definition():
    st.title("Problem Definition")
    st.write("""
    ## Defining the Healthcare Problem
    
    Before diving into the data and algorithms, it's crucial to clearly define the problem we're trying to solve.
    
    ### Our Objective:
    **To predict whether a patient has heart disease based on clinical and demographic features.**
    
    This is a **binary classification problem** - we need to classify patients into two categories:
    - 🟢 **No heart disease** (Class 0)
    - 🔴 **Has heart disease** (Class 1)
    
    ### Why This Matters:
    - Heart disease is one of the leading causes of death globally
    - Early detection can significantly improve patient outcomes
    - ML models can help identify at-risk patients who might benefit from preventive care
    
    ### Evaluation Criteria:
    Since this is a medical diagnosis problem, we need to be careful about how we evaluate our model:
    
    - **Accuracy**: Overall correctness of predictions
    - **Sensitivity (Recall)**: Ability to correctly identify patients with heart disease
    - **Specificity**: Ability to correctly identify healthy patients
    - **Precision**: What proportion of positive identifications was actually correct
    - **F1-score**: Harmonic mean of precision and recall
    
    ### Considerations in Healthcare ML:
    - **False negatives** can be very costly (missing a diagnosis)
    - **Model interpretability** is important for clinical adoption
    - **Ethical implications** of automated diagnostic tools
    """)
    
    st.info("💡 **ML Concept**: In classification problems, we need to be especially careful about class imbalance and choosing the right evaluation metrics for our specific healthcare context.")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous: Introduction"):
            st.session_state.current_step = 0
            st.rerun()
    with col2:
        if st.button("Next: Data Collection →"):
            st.session_state.current_step = 2
            st.rerun()

def show_data_collection():
    st.title("Data Collection")
    st.write("""
    ## Collecting Healthcare Data
    
    For this tutorial, we'll use the **Heart Disease dataset** from the UCI Machine Learning Repository.
    
    ### About the Dataset:
    This dataset contains medical records of patients and whether they have heart disease. 
    It's a common benchmark dataset for binary classification in healthcare.
    
    ### Dataset Features:
    - **age**: Age in years
    - **sex**: Sex (1 = male, 0 = female)
    - **cp**: Chest pain type (0-3)
    - **trestbps**: Resting blood pressure (in mm Hg)
    - **chol**: Serum cholesterol in mg/dl
    - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    - **restecg**: Resting electrocardiographic results (0-2)
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (1 = yes, 0 = no)
    - **oldpeak**: ST depression induced by exercise relative to rest
    - **slope**: The slope of the peak exercise ST segment (0-2)
    - **ca**: Number of major vessels colored by fluoroscopy (0-3)
    - **thal**: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)
    - **target**: Heart disease diagnosis (1 = present, 0 = absent)
    
    ### Data Collection in Real-World Healthcare Projects:
    In real-world healthcare ML projects, data might come from:
    - Electronic Health Records (EHR)
    - Clinical trials
    - Medical imaging repositories
    - Wearable devices
    - Patient surveys
    
    ### Data Collection Challenges in Healthcare:
    - Privacy concerns and regulations (HIPAA, GDPR)
    - Data silos across healthcare systems
    - Inconsistent formats and standards
    - Missing or incorrect data
    - Selection bias in historical medical records
    """)
    
    # Load the data when user clicks button
    if st.button("Load Heart Disease Dataset"):
        with st.spinner("Loading dataset..."):
            data = data_loader.load_heart_disease_data()
            st.session_state.data = data
            
            st.success("Dataset loaded successfully!")
            st.write("### Preview of the Heart Disease Dataset")
            st.dataframe(data.head())
            
            st.write("### Dataset Shape")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            
            st.write("### Target Distribution")
            fig = px.pie(
                names=['No Heart Disease', 'Heart Disease'],
                values=[(data['target'] == 0).sum(), (data['target'] == 1).sum()],
                title="Distribution of Heart Disease Cases",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig)
    
    # Show data if it's already loaded
    elif st.session_state.data is not None:
        st.write("### Preview of the Heart Disease Dataset")
        st.dataframe(st.session_state.data.head())
        
        st.write("### Dataset Shape")
        st.write(f"Rows: {st.session_state.data.shape[0]}, Columns: {st.session_state.data.shape[1]}")
        
        st.write("### Target Distribution")
        fig = px.pie(
            names=['No Heart Disease', 'Heart Disease'],
            values=[(st.session_state.data['target'] == 0).sum(), (st.session_state.data['target'] == 1).sum()],
            title="Distribution of Heart Disease Cases",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig)
    
    st.info("💡 **ML Concept**: Good quality data is the foundation of any machine learning project. In healthcare, you must ensure data quality while maintaining patient privacy.")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous: Problem Definition"):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.button("Next: Data Exploration →"):
            st.session_state.current_step = 3
            st.rerun()

def show_data_exploration():
    st.title("Data Exploration (EDA)")
    
    if st.session_state.data is None:
        st.warning("Please load the dataset in the 'Data Collection' step before proceeding.")
        return
    
    st.write("""
    ## Exploratory Data Analysis
    
    Exploratory Data Analysis (EDA) helps us understand the dataset before building models. 
    This step is crucial for:
    
    - Understanding data distributions and relationships
    - Identifying patterns and trends
    - Detecting outliers and anomalies
    - Formulating hypotheses about important features
    """)
    
    data = st.session_state.data
    
    # Statistical summary
    st.write("### Statistical Summary")
    st.write("This shows basic statistics like mean, standard deviation, and percentiles for each numeric feature:")
    st.dataframe(data.describe())
    
    # Check for missing values
    st.write("### Missing Value Analysis")
    missing_values = data.isnull().sum()
    if missing_values.sum() == 0:
        st.success("✅ Great! There are no missing values in the dataset.")
    else:
        st.warning(f"There are {missing_values.sum()} missing values in the dataset.")
        st.dataframe(missing_values[missing_values > 0])
    
    # Explore data distributions
    st.write("### Feature Distributions")
    
    # Let user select feature for histogram
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_feature = st.selectbox("Select a feature to view its distribution:", numeric_columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"#### Distribution of {selected_feature}")
        fig = px.histogram(
            data, 
            x=selected_feature, 
            color="target",
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            barmode="overlay",
            title=f"Distribution of {selected_feature} by Heart Disease Status",
            labels={"target": "Heart Disease", "x": selected_feature},
            opacity=0.7
        )
        fig.update_layout(legend_title_text="Heart Disease", 
                          legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig)
        
    with col2:
        st.write(f"#### Box Plot for {selected_feature}")
        fig = px.box(
            data,
            x="target",
            y=selected_feature,
            color="target",
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            title=f"Box Plot of {selected_feature} by Heart Disease Status",
            labels={"target": "Heart Disease", "y": selected_feature},
            points="all"
        )
        fig.update_layout(legend_title_text="Heart Disease", 
                          legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig)
    
    # Feature correlations
    st.write("### Feature Correlations")
    st.write("""
    Correlation analysis helps us understand relationships between variables:
    - Values close to 1 indicate strong positive correlation
    - Values close to -1 indicate strong negative correlation
    - Values close to 0 indicate little to no linear correlation
    """)
    
    corr_matrix = data.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix of Features"
    )
    st.plotly_chart(fig)
    
    # Target correlation analysis
    st.write("### Correlation with Target Variable")
    target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
    
    fig = px.bar(
        x=target_corr.index,
        y=target_corr.values,
        title="Feature Correlation with Heart Disease Target",
        labels={"x": "Features", "y": "Correlation Coefficient"},
        color=target_corr.values,
        color_continuous_scale='RdBu_r'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    
    # Advanced exploration: Pair plot for selected features
    st.write("### Advanced Exploration: Pair Plot")
    st.write("""
    Pair plots show relationships between multiple features at once. 
    Select 2-4 features to see how they relate to each other and the target.
    """)
    
    top_features = target_corr.abs().nlargest(8).index.tolist()
    selected_features = st.multiselect(
        "Select 2-4 features for the pair plot:",
        top_features,
        default=top_features[:3]
    )
    
    if len(selected_features) >= 2 and len(selected_features) <= 4:
        pair_data = data[selected_features + ['target']].copy()
        pair_data['Heart Disease'] = pair_data['target'].map({0: "No", 1: "Yes"})
        
        fig = px.scatter_matrix(
            pair_data,
            dimensions=selected_features,
            color="Heart Disease",
            color_discrete_map={"No": "#3498db", "Yes": "#e74c3c"},
            title="Pair Plot of Selected Features",
            opacity=0.7
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig)
    elif len(selected_features) > 4:
        st.warning("Please select at most 4 features for better visualization.")
    elif len(selected_features) < 2:
        st.warning("Please select at least 2 features for the pair plot.")
    
    # Key insights
    st.write("### Key Insights from EDA")
    st.write("""
    Here are some observations from our exploratory analysis:
    
    1. **Target Distribution**: The dataset is relatively balanced between patients with and without heart disease.
    
    2. **Age Patterns**: There appears to be a higher incidence of heart disease with increasing age.
    
    3. **Correlation Analysis**: Features like 'thalach' (maximum heart rate), 'cp' (chest pain type), and 'oldpeak' (ST depression) have stronger correlations with the target variable.
    
    4. **Feature Distributions**: Many features show different distributions for patients with and without heart disease, suggesting they have predictive power.
    
    5. **No Missing Values**: The dataset is complete with no missing values, which simplifies our preprocessing steps.
    """)
    
    st.info("💡 **ML Concept**: Thorough EDA helps us understand our data better and guides feature selection, preprocessing, and model selection decisions later in the workflow.")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous: Data Collection"):
            st.session_state.current_step = 2
            st.rerun()
    with col2:
        if st.button("Next: Data Preprocessing →"):
            st.session_state.current_step = 4
            st.rerun()

def show_data_preprocessing():
    st.title("Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please load the dataset in the 'Data Collection' step before proceeding.")
        return
    
    st.write("""
    ## Data Preprocessing
    
    Data preprocessing is a crucial step that transforms raw data into a format suitable for machine learning algorithms. 
    Good preprocessing can significantly improve model performance.
    
    ### Common Preprocessing Steps:
    1. **Handling missing values** - Imputation or removal
    2. **Encoding categorical variables** - Converting text to numbers
    3. **Feature scaling** - Normalizing or standardizing numerical features
    4. **Handling outliers** - Removing or transforming extreme values
    5. **Splitting data** - Creating training and testing sets
    """)
    
    data = st.session_state.data
    
    # Display preprocessing options
    st.write("### Preprocessing Options")
    
    # Handling missing values - already confirmed none in EDA, but mention it
    st.write("#### 1. Handling Missing Values")
    st.success("✅ Our dataset has no missing values, so we can skip this step!")
    
    # Encoding categorical variables
    st.write("#### 2. Encoding Categorical Variables")
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_columns) == 0:
        st.success("✅ Our dataset doesn't have categorical variables that need encoding!")
    else:
        encoding_method = st.radio(
            "Select encoding method for categorical variables:",
            ["One-Hot Encoding", "Label Encoding"]
        )
        
        if encoding_method == "One-Hot Encoding":
            st.write("One-Hot Encoding creates binary columns for each category.")
        else:
            st.write("Label Encoding assigns a unique number to each category.")
    
    # Feature scaling
    st.write("#### 3. Feature Scaling")
    scaling_method = st.radio(
        "Select scaling method for numerical features:",
        ["StandardScaler", "MinMaxScaler", "No Scaling"]
    )
    
    if scaling_method == "StandardScaler":
        st.write("""
        **StandardScaler** standardizes features by removing the mean and scaling to unit variance:
        - Mean = 0
        - Standard deviation = 1
        - Good for algorithms sensitive to feature magnitudes (like SVM, PCA)
        """)
    elif scaling_method == "MinMaxScaler":
        st.write("""
        **MinMaxScaler** scales features to a fixed range, usually [0, 1]:
        - Preserves the shape of the original distribution
        - Good when you need bounded values
        """)
    else:
        st.write("""
        **No Scaling** keeps the original values:
        - Some algorithms like tree-based models don't require scaling
        - Maintains interpretability of features
        """)
    
    # Outlier handling
    st.write("#### 4. Handling Outliers")
    outlier_method = st.radio(
        "Select method for handling outliers:",
        ["No Treatment", "Remove Outliers", "Cap Outliers"]
    )
    
    if outlier_method == "No Treatment":
        st.write("Keep all data points, including potential outliers.")
    elif outlier_method == "Remove Outliers":
        st.write("Remove data points that are more than 3 standard deviations from the mean.")
    else:
        st.write("Cap outliers at 3 standard deviations from the mean.")
    
    # Train-test split
    st.write("#### 5. Train-Test Split")
    test_size = st.slider("Test set size (%):", 10, 40, 20, 5)
    st.write(f"The dataset will be split into {100-test_size}% training and {test_size}% testing data.")
    
    # Process the data when button is clicked
    if st.button("Apply Preprocessing"):
        with st.spinner("Preprocessing data..."):
            # Apply selected preprocessing steps
            preprocessed_data, X_train, X_test, y_train, y_test = data_preprocessing.preprocess_data(
                data, 
                scaling_method=scaling_method,
                outlier_method=outlier_method,
                test_size=test_size/100,
                random_state=42
            )
            
            # Store preprocessed data in session state
            st.session_state.preprocessed_data = preprocessed_data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("Data preprocessing completed successfully!")
            
            # Show data shapes after preprocessing
            st.write("### Preprocessed Data")
            st.write(f"Training set: {X_train.shape[0]} samples")
            st.write(f"Testing set: {X_test.shape[0]} samples")
            
            # Show class distribution in train/test sets
            train_dist = pd.Series(y_train).value_counts(normalize=True) * 100
            test_dist = pd.Series(y_test).value_counts(normalize=True) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Training Set Class Distribution")
                fig = px.pie(
                    names=['No Heart Disease', 'Heart Disease'],
                    values=[train_dist[0], train_dist[1]],
                    title="Training Set Target Distribution",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                st.plotly_chart(fig)
                
            with col2:
                st.write("#### Testing Set Class Distribution")
                fig = px.pie(
                    names=['No Heart Disease', 'Heart Disease'],
                    values=[test_dist[0], test_dist[1]],
                    title="Testing Set Target Distribution",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                st.plotly_chart(fig)
            
            # Show effect of preprocessing
            if scaling_method != "No Scaling":
                st.write("### Effect of Feature Scaling")
                
                feature = st.selectbox("Select a feature to view scaling effect:", X_train.columns)
                
                # Compare original vs scaled distributions
                fig = go.Figure()
                
                # Original data distribution
                original_feature = data[feature]
                fig.add_trace(go.Histogram(
                    x=original_feature,
                    name="Original Data",
                    opacity=0.7,
                    marker_color='#3498db'
                ))
                
                # Scaled data distribution
                scaled_feature = X_train[feature]
                fig.add_trace(go.Histogram(
                    x=scaled_feature,
                    name="Scaled Data",
                    opacity=0.7,
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    title=f"Distribution of {feature}: Original vs Scaled",
                    xaxis_title=feature,
                    yaxis_title="Count",
                    barmode='overlay',
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                )
                
                st.plotly_chart(fig)
                
                st.write(f"""
                **Before scaling:**
                - Min: {original_feature.min():.4f}
                - Max: {original_feature.max():.4f}
                - Mean: {original_feature.mean():.4f}
                - Std: {original_feature.std():.4f}
                
                **After {scaling_method}:**
                - Min: {scaled_feature.min():.4f}
                - Max: {scaled_feature.max():.4f}
                - Mean: {scaled_feature.mean():.4f}
                - Std: {scaled_feature.std():.4f}
                """)
    
    # Show preprocessing results if already processed
    elif st.session_state.preprocessed_data is not None:
        st.success("Data has already been preprocessed!")
        
        # Show data shapes after preprocessing
        st.write("### Preprocessed Data")
        st.write(f"Training set: {st.session_state.X_train.shape[0]} samples")
        st.write(f"Testing set: {st.session_state.X_test.shape[0]} samples")
        
        # Show class distribution in train/test sets
        train_dist = pd.Series(st.session_state.y_train).value_counts(normalize=True) * 100
        test_dist = pd.Series(st.session_state.y_test).value_counts(normalize=True) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Training Set Class Distribution")
            fig = px.pie(
                names=['No Heart Disease', 'Heart Disease'],
                values=[train_dist[0], train_dist[1]],
                title="Training Set Target Distribution",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig)
            
        with col2:
            st.write("#### Testing Set Class Distribution")
            fig = px.pie(
                names=['No Heart Disease', 'Heart Disease'],
                values=[test_dist[0], test_dist[1]],
                title="Testing Set Target Distribution",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig)
    
    st.info("""
    💡 **ML Concept**: Proper preprocessing ensures that:
    
    1. Data is in the right format for algorithms
    2. Training and testing distributions are similar
    3. Model assumptions about data are satisfied
    4. We avoid leaking information from test to training data
    """)
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous: Data Exploration"):
            st.session_state.current_step = 3
            st.rerun()
    with col2:
        if st.button("Next: Feature Engineering →"):
            st.session_state.current_step = 5
            st.rerun()

def show_feature_engineering():
    st.title("Feature Engineering")
    
    if st.session_state.preprocessed_data is None:
        st.warning("Please complete the data preprocessing step before proceeding.")
        return
    
    st.write("""
    ## Feature Engineering
    
    Feature engineering is the process of creating new features or transforming existing ones to improve model performance.
    It's often considered one of the most creative and impactful parts of the ML workflow.
    
    ### Common Feature Engineering Techniques:
    1. **Feature Selection** - Choosing the most relevant features
    2. **Feature Creation** - Creating new features from existing ones
    3. **Feature Transformation** - Applying mathematical functions to features
    4. **Dimensionality Reduction** - Reducing the number of features
    """)
    
    # Get preprocessed data
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    
    # Feature selection options
    st.write("### 1. Feature Selection")
    selection_method = st.radio(
        "Select a feature selection method:",
        ["Statistical Tests", "Model-based Selection", "Manual Selection"]
    )
    
    if selection_method == "Statistical Tests":
        st.write("""
        **Statistical Tests** measure the relationship between each feature and the target:
        - Higher scores indicate stronger relationships
        - Commonly used tests: chi-squared, ANOVA, correlation
        """)
        
        # Compute feature importance using Chi-squared test
        importances = feature_engineering.calculate_feature_importance(X_train, y_train, method='chi2')
        
        # Display feature importance
        fig = px.bar(
            x=importances.index,
            y=importances.values,
            title="Feature Importance (Chi-squared Test)",
            labels={"x": "Features", "y": "Importance Score"},
            color=importances.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
        
        # Let user choose number of features to select
        num_features = st.slider("Select top N features:", 1, len(importances), min(8, len(importances)))
        selected_features = importances.nlargest(num_features).index.tolist()
        
    elif selection_method == "Model-based Selection":
        st.write("""
        **Model-based Selection** uses a machine learning model to determine feature importance:
        - Tree-based models (like Random Forest) provide feature importance scores
        - Models can capture non-linear relationships between features and target
        """)
        
        # Compute feature importance using Random Forest
        importances = feature_engineering.calculate_feature_importance(X_train, y_train, method='model')
        
        # Display feature importance
        fig = px.bar(
            x=importances.index,
            y=importances.values,
            title="Feature Importance (Random Forest)",
            labels={"x": "Features", "y": "Importance Score"},
            color=importances.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
        
        # Let user choose number of features to select
        num_features = st.slider("Select top N features:", 1, len(importances), min(8, len(importances)))
        selected_features = importances.nlargest(num_features).index.tolist()
        
    else:  # Manual Selection
        st.write("""
        **Manual Selection** allows you to choose features based on domain knowledge or preference:
        - Useful when you have specific hypotheses to test
        - Provides full control over the feature set
        """)
        
        # Let user manually select features
        selected_features = st.multiselect(
            "Select features to include in the model:",
            X_train.columns.tolist(),
            default=X_train.columns.tolist()[:5]
        )
    
    # Feature creation and transformation
    st.write("### 2. Feature Creation & Transformation")
    st.write("""
    In healthcare models, creating interactions between features can capture important relationships:
    - Age and cholesterol might interact to affect heart disease risk
    - Multiple risk factors might have compound effects
    """)
    
    create_interactions = st.checkbox("Create interaction features (pairs of selected features)")
    
    if create_interactions and len(selected_features) >= 2:
        st.write("This will create new features by multiplying pairs of selected features.")
        
        # Calculate potential number of interaction features
        num_interactions = len(selected_features) * (len(selected_features) - 1) // 2
        st.write(f"Potential number of interaction features: {num_interactions}")
        
        # Let user limit the number of interactions
        max_interactions = st.slider(
            "Maximum number of interactions to create:", 
            1, 
            num_interactions, 
            min(5, num_interactions)
        )
    else:
        max_interactions = 0
    
    # Polynomial features
    create_polynomial = st.checkbox("Create polynomial features (squared terms)")
    
    if create_polynomial:
        st.write("This will create squared terms for each selected feature.")
    
    # Apply feature engineering when button is clicked
    if st.button("Apply Feature Engineering"):
        with st.spinner("Engineering features..."):
            # Apply feature engineering
            X_train_new, X_test_new, feature_names = feature_engineering.engineer_features(
                st.session_state.X_train,
                st.session_state.X_test,
                selected_features=selected_features,
                create_interactions=create_interactions,
                max_interactions=max_interactions,
                create_polynomial=create_polynomial
            )
            
            # Update session state
            st.session_state.X_train = X_train_new
            st.session_state.X_test = X_test_new
            st.session_state.features_selected = feature_names
            
            st.success("Feature engineering completed successfully!")
            
            # Show new feature set
            st.write("### New Feature Set")
            st.write(f"Number of features: {len(feature_names)}")
            st.write("Feature names:")
            
            # Display feature names in columns
            cols = st.columns(3)
            for i, feature in enumerate(feature_names):
                cols[i % 3].write(f"- {feature}")
            
            # Show correlation matrix of new features
            st.write("### Correlation Matrix of New Features")
            corr_matrix = X_train_new.corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix of Engineered Features"
            )
            st.plotly_chart(fig)
    
    # Show results if feature engineering has already been applied
    elif st.session_state.features_selected is not None:
        st.success("Feature engineering has already been applied!")
        
        # Show new feature set
        st.write("### Current Feature Set")
        st.write(f"Number of features: {len(st.session_state.features_selected)}")
        st.write("Feature names:")
        
        # Display feature names in columns
        cols = st.columns(3)
        for i, feature in enumerate(st.session_state.features_selected):
            cols[i % 3].write(f"- {feature}")
        
        # Show correlation matrix of new features
        st.write("### Correlation Matrix of Current Features")
        corr_matrix = st.session_state.X_train.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of Features"
        )
        st.plotly_chart(fig)
    
    st.info("""
    💡 **ML Concept**: Feature engineering can often have a greater impact on model performance than algorithm selection:
    
    1. It helps incorporate domain knowledge into the model
    2. It can reveal hidden patterns that models might miss
    3. It can make nonlinear relationships more accessible to linear models
    4. It reduces the complexity needed in the model architecture
    """)
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous: Data Preprocessing"):
            st.session_state.current_step = 4
            st.rerun()
    with col2:
        if st.button("Next: Model Selection & Training →"):
            st.session_state.current_step = 6
            st.rerun()

def show_model_training():
    st.title("Model Selection & Training")
    
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Please complete the previous steps before proceeding.")
        return
    
    st.write("""
    ## Model Selection & Training
    
    Choosing the right model and training it effectively are crucial steps in the ML workflow.
    Different algorithms have different strengths, weaknesses, and assumptions.
    
    ### Common ML Algorithms for Classification:
    1. **Logistic Regression** - Simple, interpretable linear model
    2. **Decision Trees** - Rule-based model with clear decision paths
    3. **Random Forest** - Ensemble of trees with better generalization
    4. **Support Vector Machines** - Powerful for separating classes with a margin
    5. **K-Nearest Neighbors** - Classification based on similar examples
    6. **Gradient Boosting** - Sequential ensemble method with high performance
    """)
    
    # Model selection
    st.write("### Model Selection")
    model_type = st.selectbox(
        "Select a machine learning algorithm:",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Gradient Boosting"]
    )
    
    # Display model description
    if model_type == "Logistic Regression":
        st.write("""
        **Logistic Regression** is a linear model for classification:
        
        **Strengths:**
        - Simple and interpretable
        - Works well for linearly separable data
        - Provides probability estimates
        - Less prone to overfitting on small datasets
        
        **Weaknesses:**
        - Cannot capture complex non-linear relationships
        - May underperform if features are highly correlated
        
        **Hyperparameters to tune:**
        - C: Regularization strength (inverse)
        - Penalty: Type of regularization (L1, L2)
        """)
        
        # Hyperparameter options
        c_value = st.slider("Regularization strength (C):", 0.01, 10.0, 1.0, 0.01)
        penalty = st.radio("Penalty type:", ["l2", "l1", "none"])
        max_iter = st.slider("Maximum iterations:", 100, 1000, 100, 100)
        
        model_params = {
            "C": c_value,
            "penalty": penalty,
            "max_iter": max_iter,
            "random_state": 42
        }
        
    elif model_type == "Decision Tree":
        st.write("""
        **Decision Tree** creates a model that predicts by learning simple decision rules:
        
        **Strengths:**
        - Easy to understand and interpret
        - Requires little data preprocessing
        - Can handle both numerical and categorical data
        - Captures non-linear patterns
        
        **Weaknesses:**
        - Can create overly complex trees that don't generalize well
        - Unstable (small variations in data can result in a different tree)
        
        **Hyperparameters to tune:**
        - Max depth: Controls tree complexity
        - Min samples split: Minimum samples required to split a node
        - Min samples leaf: Minimum samples required in a leaf node
        """)
        
        # Hyperparameter options
        max_depth = st.slider("Maximum depth:", 1, 20, 5, 1)
        min_samples_split = st.slider("Minimum samples to split:", 2, 20, 2, 1)
        min_samples_leaf = st.slider("Minimum samples in leaf:", 1, 20, 1, 1)
        
        model_params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": 42
        }
        
    elif model_type == "Random Forest":
        st.write("""
        **Random Forest** is an ensemble of decision trees:
        
        **Strengths:**
        - Better accuracy than single decision trees
        - Less prone to overfitting
        - Handles large datasets efficiently
        - Provides feature importance estimates
        
        **Weaknesses:**
        - Less interpretable than a single decision tree
        - Slower to train and predict than simpler models
        
        **Hyperparameters to tune:**
        - N estimators: Number of trees in the forest
        - Max depth: Maximum depth of the trees
        - Min samples split: Minimum samples required to split a node
        """)
        
        # Hyperparameter options
        n_estimators = st.slider("Number of trees:", 10, 200, 100, 10)
        max_depth = st.slider("Maximum depth:", 1, 20, 10, 1)
        min_samples_split = st.slider("Minimum samples to split:", 2, 20, 2, 1)
        
        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": 42
        }
        
    elif model_type == "Support Vector Machine":
        st.write("""
        **Support Vector Machine (SVM)** finds the optimal boundary between classes:
        
        **Strengths:**
        - Effective in high-dimensional spaces
        - Versatile (different kernel functions for various decision boundaries)
        - Memory efficient as it uses a subset of training points
        
        **Weaknesses:**
        - Does not provide probability estimates directly
        - Can be slow to train on large datasets
        - Sensitive to feature scaling
        
        **Hyperparameters to tune:**
        - C: Regularization parameter
        - Kernel: Type of kernel function
        - Gamma: Kernel coefficient (for 'rbf', 'poly' and 'sigmoid')
        """)
        
        # Hyperparameter options
        c_value = st.slider("Regularization parameter (C):", 0.1, 10.0, 1.0, 0.1)
        kernel = st.radio("Kernel type:", ["linear", "rbf", "poly"])
        gamma = st.radio("Gamma:", ["scale", "auto"])
        
        model_params = {
            "C": c_value,
            "kernel": kernel,
            "gamma": gamma,
            "probability": True,
            "random_state": 42
        }
        
    elif model_type == "K-Nearest Neighbors":
        st.write("""
        **K-Nearest Neighbors (KNN)** classifies based on the most similar training examples:
        
        **Strengths:**
        - Simple and intuitive
        - No training phase (lazy learning)
        - Naturally handles multi-class problems
        - Effective for many applications
        
        **Weaknesses:**
        - Computationally intensive during prediction
        - Sensitive to irrelevant features
        - Requires feature scaling
        
        **Hyperparameters to tune:**
        - n_neighbors: Number of neighbors to consider
        - weights: Weight function used in prediction
        - p: Power parameter for Minkowski distance
        """)
        
        # Hyperparameter options
        n_neighbors = st.slider("Number of neighbors:", 1, 20, 5, 1)
        weights = st.radio("Weight function:", ["uniform", "distance"])
        p = st.radio("Minkowski power parameter:", [1, 2])
        
        model_params = {
            "n_neighbors": n_neighbors,
            "weights": weights,
            "p": p
        }
        
    else:  # Gradient Boosting
        st.write("""
        **Gradient Boosting** builds an ensemble of trees sequentially, each correcting errors of the previous ones:
        
        **Strengths:**
        - Often provides best-in-class accuracy
        - Handles different types of data
        - Naturally handles missing values
        - Less preprocessing required
        
        **Weaknesses:**
        - More parameters to tune
        - Prone to overfitting
        - Computationally intensive
        
        **Hyperparameters to tune:**
        - n_estimators: Number of boosting stages
        - learning_rate: Shrinks contribution of each tree
        - max_depth: Maximum depth of each tree
        """)
        
        # Hyperparameter options
        n_estimators = st.slider("Number of boosting stages:", 10, 200, 100, 10)
        learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01)
        max_depth = st.slider("Maximum depth:", 1, 10, 3, 1)
        
        model_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": 42
        }
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Train the model
            model, training_time = model_training.train_model(
                st.session_state.X_train,
                st.session_state.y_train,
                model_type=model_type,
                model_params=model_params
            )
            
            # Store model in session state
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.model_params = model_params
            
            st.success(f"Model trained successfully in {training_time:.4f} seconds!")
            
            # Make predictions on training data to show learning
            y_train_pred = model.predict(st.session_state.X_train)
            train_accuracy = (y_train_pred == st.session_state.y_train).mean() * 100
            
            st.write(f"Training accuracy: {train_accuracy:.2f}%")
            
            # Display model information
            st.write("### Model Information")
            
            if model_type == "Logistic Regression":
                # Show coefficients for top features
                coefficients = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Coefficient': model.coef_[0]
                }).sort_values('Coefficient', ascending=False)
                
                st.write("#### Feature Coefficients")
                st.write("Positive coefficients increase the probability of heart disease, negative ones decrease it.")
                
                fig = px.bar(
                    coefficients,
                    x='Feature',
                    y='Coefficient',
                    title="Logistic Regression Coefficients",
                    color='Coefficient',
                    color_continuous_scale='RdBu_r'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
                
            elif model_type == "Decision Tree" or model_type == "Random Forest" or model_type == "Gradient Boosting":
                # Show feature importances
                importances = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("#### Feature Importances")
                
                fig = px.bar(
                    importances,
                    x='Feature',
                    y='Importance',
                    title=f"{model_type} Feature Importances",
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
                
                # For Decision Tree, offer to visualize the tree
                if model_type == "Decision Tree" and st.session_state.X_train.shape[1] < 10:
                    st.write("#### Decision Tree Visualization")
                    st.write("Note: This visualization is simplified and may not show the complete tree.")
                    
                    tree_viz = utils.get_decision_tree_viz(model, st.session_state.X_train.columns)
                    st.graphviz_chart(tree_viz)
            
            # For KNN, show effect of K on a sample
            elif model_type == "K-Nearest Neighbors":
                st.write("#### K-Nearest Neighbors Visualization")
                st.write("This shows how different values of K would classify a sample:")
                
                # Get a sample data point
                sample_idx = np.random.randint(0, len(st.session_state.X_test))
                sample = st.session_state.X_test.iloc[sample_idx:sample_idx+1]
                true_label = st.session_state.y_test.iloc[sample_idx]
                
                # Calculate and display predictions for different K values
                k_values = list(range(1, 11))
                k_predictions = []
                
                for k in k_values:
                    temp_model = model_training.train_model(
                        st.session_state.X_train,
                        st.session_state.y_train,
                        model_type="K-Nearest Neighbors",
                        model_params={"n_neighbors": k, "weights": "uniform"}
                    )[0]
                    k_predictions.append(temp_model.predict(sample)[0])
                
                # Create a plot
                fig = px.line(
                    x=k_values,
                    y=k_predictions,
                    title=f"KNN Predictions for Different K Values (True Label: {true_label})",
                    labels={"x": "Number of Neighbors (K)", "y": "Prediction"},
                    markers=True
                )
                
                # Add horizontal line for true label
                fig.add_hline(
                    y=true_label,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"True Label: {true_label}",
                    annotation_position="bottom right"
                )
                
                # Add vertical line for selected K
                fig.add_vline(
                    x=model_params["n_neighbors"],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Selected K: {model_params['n_neighbors']}",
                    annotation_position="top right"
                )
                
                st.plotly_chart(fig)
    
    # Show model results if a model has been trained
    elif st.session_state.model is not None:
        st.success(f"A {st.session_state.model_type} model has already been trained!")
        
        # Make predictions on training data to show learning
        y_train_pred = st.session_state.model.predict(st.session_state.X_train)
        train_accuracy = (y_train_pred == st.session_state.y_train).mean() * 100
        
        st.write(f"Training accuracy: {train_accuracy:.2f}%")
        
        # Display model information
        st.write("### Model Information")
        
        if st.session_state.model_type == "Logistic Regression":
            # Show coefficients for top features
            coefficients = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Coefficient': st.session_state.model.coef_[0]
            }).sort_values('Coefficient', ascending=False)
            
            st.write("#### Feature Coefficients")
            st.write("Positive coefficients increase the probability of heart disease, negative ones decrease it.")
            
            fig = px.bar(
                coefficients,
                x='Feature',
                y='Coefficient',
                title="Logistic Regression Coefficients",
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
            
        elif st.session_state.model_type in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
            # Show feature importances
            importances = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.write("#### Feature Importances")
            
            fig = px.bar(
                importances,
                x='Feature',
                y='Importance',
                title=f"{st.session_state.model_type} Feature Importances",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
    
    st.info("""
    💡 **ML Concept**: Model selection should consider:
    
    1. The nature of your data (linear vs. non-linear relationships)
    2. Interpretability requirements (especially important in healthcare)
    3. Dataset size (complex models need more data)
    4. Computational constraints (training and inference time)
    5. The balance between bias and variance
    """)
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous: Feature Engineering"):
            st.session_state.current_step = 5
            st.rerun()
    with col2:
        if st.button("Next: Model Evaluation →"):
            st.session_state.current_step = 7
            st.rerun()

def show_model_evaluation():
    st.title("Model Evaluation")
    
    if st.session_state.model is None:
        st.warning("Please train a model in the previous step before proceeding.")
        return
    
    st.write("""
    ## Model Evaluation
    
    Proper evaluation is crucial to understand how well your model will perform on new, unseen data.
    This is especially important in healthcare, where the consequences of incorrect predictions can be serious.
    
    ### Common Evaluation Metrics for Classification:
    1. **Accuracy** - Overall correctness (correct predictions / total predictions)
    2. **Precision** - When model predicts positive, how often is it correct? (TP / (TP + FP))
    3. **Recall (Sensitivity)** - What proportion of actual positives is identified correctly? (TP / (TP + FN))
    4. **F1-Score** - Harmonic mean of precision and recall
    5. **ROC Curve & AUC** - Performance across all classification thresholds
    """)
    
    # Get necessary data from session state
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    model_type = st.session_state.model_type
    
    # Evaluate the model on test data
    with st.spinner("Evaluating model on test data..."):
        results = model_evaluation.evaluate_model(model, X_test, y_test)
        
        # Store predictions in session state
        st.session_state.predictions = results["y_pred"]
        st.session_state.probabilities = results["y_prob"]
        
        # Display classification report metrics
        st.write("### Classification Metrics")
        
        # Create a metrics dashboard with 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{results['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{results['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{results['f1']:.2%}")
        
        # Explain the metrics in the healthcare context
        st.write("""
        **In the context of heart disease prediction:**
        
        - **Accuracy**: Overall percentage of patients correctly classified
        - **Precision**: When we predict a patient has heart disease, how often are we right?
        - **Recall**: What percentage of patients with heart disease did we correctly identify?
        - **F1-Score**: Balance between precision and recall
        
        In healthcare, high recall (sensitivity) is often prioritized to avoid missing patients with the disease (false negatives).
        """)
        
        # Confusion Matrix
        st.write("### Confusion Matrix")
        st.write("""
        The confusion matrix shows:
        - **True Positives (TP)**: Correctly predicted heart disease
        - **True Negatives (TN)**: Correctly predicted no heart disease
        - **False Positives (FP)**: Incorrectly predicted heart disease (Type I error)
        - **False Negatives (FN)**: Incorrectly predicted no heart disease (Type II error)
        """)
        
        fig = px.imshow(
            results["conf_matrix"],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["No Disease", "Heart Disease"],
            y=["No Disease", "Heart Disease"],
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        st.plotly_chart(fig)
        
        # ROC Curve and AUC
        st.write("### ROC Curve & AUC")
        st.write("""
        The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate 
        at different classification thresholds.
        
        The Area Under the Curve (AUC) measures the model's ability to distinguish between classes.
        - AUC = 1.0: Perfect classifier
        - AUC = 0.5: No better than random guessing
        """)
        
        fig = px.line(
            x=results["fpr"],
            y=results["tpr"],
            title=f"ROC Curve (AUC = {results['auc']:.3f})",
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
            width=700,
            height=500
        )
        
        # Add the diagonal line representing random guessing
        fig.add_shape(
            type='line',
            line=dict(dash='dash', color='gray'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        st.plotly_chart(fig)
        
        # Prediction Threshold Analysis
        st.write("### Prediction Threshold Analysis")
        st.write("""
        Classification models output a probability that is converted to a class based on a threshold (default 0.5).
        Adjusting this threshold allows balancing between precision and recall.
        """)
        
        # Interactive threshold selector
        threshold = st.slider(
            "Prediction probability threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Calculate metrics at the selected threshold
        custom_results = model_evaluation.evaluate_at_threshold(
            results["y_prob"],
            y_test,
            threshold
        )
        
        # Display metrics for the custom threshold
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{custom_results['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{custom_results['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{custom_results['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{custom_results['f1']:.2%}")
        
        # Precision-Recall curve
        st.write("### Precision-Recall Curve")
        st.write("""
        The Precision-Recall curve shows the trade-off between precision and recall at different thresholds.
        In medical applications, this curve is often more informative than the ROC curve when classes are imbalanced.
        """)
        
        fig = px.line(
            x=results["recalls"],
            y=results["precisions"],
            title="Precision-Recall Curve",
            labels={"x": "Recall", "y": "Precision"},
            width=700,
            height=500
        )
        
        # Mark the current threshold on the curve
        current_point_index = (np.abs(results["thresholds"] - threshold)).argmin()
        fig.add_trace(
            go.Scatter(
                x=[results["recalls"][current_point_index]],
                y=[results["precisions"][current_point_index]],
                mode="markers",
                marker=dict(size=10, color="red"),
                name=f"Threshold = {threshold:.2f}"
            )
        )
        
        st.plotly_chart(fig)
        
        # Feature importance and model interpretation
        st.write("### Model Interpretation")
        
        if model_type == "Logistic Regression":
            # Show coefficients for features
            coefficients = pd.DataFrame({
                'Feature': X_test.columns,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', ascending=False)
            
            st.write("#### Feature Influence on Heart Disease Prediction")
            st.write("""
            For logistic regression, the coefficients indicate how each feature influences the prediction:
            - Positive coefficients increase the probability of heart disease
            - Negative coefficients decrease the probability of heart disease
            - Larger absolute values have stronger influence
            """)
            
            fig = px.bar(
                coefficients,
                x='Feature',
                y='Coefficient',
                title="Feature Influence in Logistic Regression",
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
            
        elif model_type in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
            # Show feature importances
            importances = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.write("#### Feature Importance in Prediction")
            st.write("""
            Feature importance shows which features have the most impact on the model's predictions:
            - Higher values indicate more important features
            - These help understand what factors most strongly influence heart disease prediction
            """)
            
            fig = px.bar(
                importances,
                x='Feature',
                y='Importance',
                title=f"Feature Importance in {model_type}",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        
        # Individual prediction exploration
        st.write("### Explore Individual Predictions")
        st.write("""
        Examine how the model makes predictions for individual patients. 
        This helps understand the model's decision-making process.
        """)
        
        # Let user select a sample from the test set
        sample_index = st.slider(
            "Select a patient from the test set:",
            0,
            len(X_test) - 1,
            0
        )
        
        # Get the selected sample
        sample = X_test.iloc[sample_index:sample_index+1]
        true_label = y_test.iloc[sample_index]
        prediction = results["y_pred"][sample_index]
        probability = results["y_prob"][sample_index]
        
        # Display sample info
        st.write(f"**True condition:** {'Heart Disease' if true_label == 1 else 'No Heart Disease'}")
        st.write(f"**Model prediction:** {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        st.write(f"**Prediction probability:** {probability[1]:.2%} chance of heart disease")
        
        # Prediction correctness
        if prediction == true_label:
            st.success("✅ Correct prediction!")
        else:
            st.error("❌ Incorrect prediction!")
        
        # Display feature values for this sample
        st.write("#### Patient Features")
        
        # Convert the sample to a more readable format
        readable_sample = pd.DataFrame({
            'Feature': sample.columns,
            'Value': sample.values[0]
        })
        
        # For better visualization, show as a horizontal bar chart
        fig = px.bar(
            readable_sample,
            y='Feature',
            x='Value',
            orientation='h',
            title="Feature Values for Selected Patient",
            color='Value',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig)
        
        if model_type in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
            st.write("#### Feature Contributions to Prediction")
            
            # Show simplified SHAP-like feature contributions for easier interpretation
            # (This is a simplified approximation since we're not using the SHAP library)
            contributions = pd.DataFrame({
                'Feature': X_test.columns,
                'Contribution': model.feature_importances_ * sample.values[0]
            }).sort_values('Contribution', ascending=False)
            
            fig = px.bar(
                contributions,
                x='Contribution',
                y='Feature',
                orientation='h',
                title="Approximate Feature Contributions to Prediction",
                color='Contribution',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig)
            
            st.write("""
            **Note:** This is a simplified approximation of feature contributions. 
            For more accurate feature importance for individual predictions, advanced tools like SHAP would be used.
            """)
    
    st.info("""
    💡 **ML Concept**: Model evaluation helps answer questions like:
    
    1. How well will the model perform on new data?
    2. Which types of errors is the model making?
    3. Which features are most important for predictions?
    4. How should we set the decision threshold for optimal performance?
    
    These insights are crucial for deploying models in healthcare settings where errors can have serious consequences.
    """)
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous: Model Training"):
            st.session_state.current_step = 6
            st.rerun()
    with col2:
        if st.button("Next: Conclusion →"):
            st.session_state.current_step = 8
            st.rerun()

def show_conclusion():
    st.title("Conclusion")
    
    st.write("""
    ## Congratulations!
    
    You've successfully completed a full machine learning workflow for heart disease prediction! 🎉
    
    ### What You've Learned:
    
    1. **Problem Definition**: Formulated a heart disease prediction problem with clear objectives
    
    2. **Data Collection**: Gathered and understood the heart disease dataset
    
    3. **Exploratory Data Analysis**: Analyzed the distributions and relationships in the data
    
    4. **Data Preprocessing**: Cleaned and prepared the data for machine learning
    
    5. **Feature Engineering**: Selected and created features to improve model performance
    
    6. **Model Selection & Training**: Chose and trained an appropriate model
    
    7. **Model Evaluation**: Assessed model performance using appropriate metrics
    
    ### Key Takeaways for Healthcare ML Projects:
    
    - **Data Quality is Crucial**: Healthcare data often has unique challenges (missing values, biases, outliers)
    
    - **Feature Engineering Matters**: Domain knowledge helps create meaningful features
    
    - **Model Interpretability**: In healthcare, being able to explain predictions is often as important as accuracy
    
    - **Evaluation Metrics**: Consider the specific healthcare context when choosing which metrics to optimize
    
    - **Ethical Considerations**: Remember that ML models in healthcare impact real patients
    """)
    
    st.write("### Your Model Performance Summary")
    
    if st.session_state.model is not None:
        # Get evaluation metrics from the most recent evaluation
        model_type = st.session_state.model_type
        
        # Generate results again to display the summary
        results = model_evaluation.evaluate_model(
            st.session_state.model,
            st.session_state.X_test,
            st.session_state.y_test
        )
        
        # Display metrics in a table
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Value': [
                f"{results['accuracy']:.2%}",
                f"{results['precision']:.2%}",
                f"{results['recall']:.2%}",
                f"{results['f1']:.2%}",
                f"{results['auc']:.3f}"
            ]
        })
        
        st.dataframe(metrics_df, hide_index=True)
        
        # Show confusion matrix
        fig = px.imshow(
            results["conf_matrix"],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["No Disease", "Heart Disease"],
            y=["No Disease", "Heart Disease"],
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        st.plotly_chart(fig)
        
        # Show feature importance or coefficients
        st.write("### Most Important Features")
        
        if model_type == "Logistic Regression":
            # Show coefficients for top features
            coefficients = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Coefficient': st.session_state.model.coef_[0]
            }).sort_values(by='Coefficient', key=abs, ascending=False).head(5)
            
            fig = px.bar(
                coefficients,
                x='Feature',
                y='Coefficient',
                title="Top Feature Coefficients",
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig)
            
        elif model_type in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
            # Show feature importances
            importances = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            fig = px.bar(
                importances,
                x='Feature',
                y='Importance',
                title="Top Feature Importances",
                color='Importance',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig)
    
    else:
        st.warning("No model has been trained yet. Complete the previous steps to see your model performance summary.")
    
    st.write("""
    ### Next Steps in Your ML Journey:
    
    1. **Try Different Algorithms**: Experiment with other models to see if they perform better
    
    2. **More Feature Engineering**: Create more complex features or try feature selection methods
    
    3. **Hyperparameter Tuning**: Fine-tune model parameters for optimal performance
    
    4. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
    
    5. **Model Deployment**: Learn how to deploy ML models in healthcare settings
    
    6. **Advanced Topics**: Explore ensemble methods, deep learning, or explainable AI
    
    ### Remember:
    
    Machine learning in healthcare is a powerful tool, but it's most effective when combined with domain expertise and careful implementation. Always consider the ethical implications and ensure models are thoroughly validated before using them in real-world healthcare applications.
    
    Thank you for completing this tutorial! We hope it helps you in your machine learning journey.
    """)
    
    # Reset button
    if st.button("Start Over"):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.current_step = 0
        st.rerun()
    
    # Previous button
    if st.button("← Previous: Model Evaluation"):
        st.session_state.current_step = 7
        st.rerun()

# Run the main application
if __name__ == "__main__":
    main()
