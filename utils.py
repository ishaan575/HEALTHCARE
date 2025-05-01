import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import graphviz
import io
import base64

def get_decision_tree_viz(model, feature_names, max_depth=3):
    """
    Generate a visualization of a decision tree model.
    
    Args:
        model: Trained decision tree model
        feature_names (list): List of feature names
        max_depth (int): Maximum depth of the tree to display
        
    Returns:
        graphviz.Source: Visualization of the decision tree
    """
    # Create a dot file for the tree
    dot_data = tree.export_graphviz(
        model,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=['No Disease', 'Heart Disease'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    # Create and return the visualization
    graph = graphviz.Source(dot_data)
    
    return graph

def create_feature_summary(data):
    """
    Create a summary of features in the dataset.
    
    Args:
        data (pandas.DataFrame): The dataset
        
    Returns:
        pandas.DataFrame: Summary of features
    """
    # Initialize lists to store feature information
    features = []
    types = []
    missing = []
    uniques = []
    descriptions = []
    
    # Iterate through each column in the dataset
    for col in data.columns:
        features.append(col)
        types.append(data[col].dtype)
        missing.append(data[col].isnull().sum())
        uniques.append(data[col].nunique())
        
        # Add a basic description based on the column name
        if col == 'age':
            descriptions.append('Age of patient in years')
        elif col == 'sex':
            descriptions.append('Gender (1 = male, 0 = female)')
        elif col == 'chest_pain_type':
            descriptions.append('Type of chest pain (0-3)')
        elif col == 'resting_bp':
            descriptions.append('Resting blood pressure (mm Hg)')
        elif col == 'cholesterol':
            descriptions.append('Serum cholesterol (mg/dl)')
        elif col == 'fasting_blood_sugar':
            descriptions.append('Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)')
        elif col == 'resting_ecg':
            descriptions.append('Resting electrocardiographic results (0-2)')
        elif col == 'max_heart_rate':
            descriptions.append('Maximum heart rate achieved')
        elif col == 'exercise_angina':
            descriptions.append('Exercise induced angina (1 = yes, 0 = no)')
        elif col == 'st_depression':
            descriptions.append('ST depression induced by exercise relative to rest')
        elif col == 'st_slope':
            descriptions.append('Slope of the peak exercise ST segment (0-2)')
        elif col == 'num_major_vessels':
            descriptions.append('Number of major vessels colored by fluoroscopy (0-3)')
        elif col == 'thalassemia':
            descriptions.append('Thalassemia (0-2)')
        elif col == 'target':
            descriptions.append('Presence of heart disease (1 = yes, 0 = no)')
        else:
            descriptions.append('No description available')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz
import io
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def get_decision_tree_viz(tree_model, feature_names, class_names=['No Heart Disease', 'Heart Disease'], max_depth=3):
    """
    Create a visualization of a decision tree model.
    
    Args:
        tree_model: Trained decision tree model
        feature_names (list): Names of the features
        class_names (list): Names of the target classes
        max_depth (int): Maximum depth of the tree to visualize
        
    Returns:
        graphviz.Source: Graphviz visualization of the tree
    """
    # Export the decision tree to DOT format
    dot_data = io.StringIO()
    export_graphviz(
        tree_model,
        out_file=dot_data,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth
    )
    
    # Create and return the graph
    graph = graphviz.Source(dot_data.getvalue())
    return graph

def format_feature_name(feature_name):
    """
    Format feature names for better readability in visualizations.
    
    Args:
        feature_name (str): Original feature name
        
    Returns:
        str: Formatted feature name
    """
    # Replace underscores with spaces
    formatted = feature_name.replace('_', ' ')
    
    # Capitalize words
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    # Handle special abbreviations
    abbreviations = ['bp', 'ecg', 'st']
    for abbr in abbreviations:
        # Replace the capitalized abbreviation with uppercase
        formatted = formatted.replace(f" {abbr.capitalize()} ", f" {abbr.upper()} ")
        # Also handle if it's at the start of the string
        if formatted.startswith(f"{abbr.capitalize()} "):
            formatted = f"{abbr.upper()} " + formatted[len(abbr)+1:]
    
    return formatted

def create_radar_chart(sample_data, feature_means, feature_names=None):
    """
    Create a radar chart comparing a sample with the population means.
    
    Args:
        sample_data (pandas.Series): Data for the sample
        feature_means (pandas.Series): Mean values for the population
        feature_names (list): Names of the features to include
        
    Returns:
        plotly.graph_objects.Figure: Radar chart
    """
    if feature_names is None:
        feature_names = sample_data.index.tolist()
    
    # Format feature names for better readability
    formatted_names = [format_feature_name(name) for name in feature_names]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add sample data
    fig.add_trace(go.Scatterpolar(
        r=[sample_data[feat] for feat in feature_names],
        theta=formatted_names,
        fill='toself',
        name='Selected Patient',
        line_color='rgba(255, 65, 54, 0.8)'
    ))
    
    # Add population means
    fig.add_trace(go.Scatterpolar(
        r=[feature_means[feat] for feat in feature_names],
        theta=formatted_names,
        fill='toself',
        name='Population Average',
        line_color='rgba(49, 130, 189, 0.8)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.5 * max([
                    max([sample_data[feat] for feat in feature_names]),
                    max([feature_means[feat] for feat in feature_names])
                ])]
            )
        ),
        title="Patient vs. Population Average",
        showlegend=True
    )
    
    return fig

def calculate_health_risk_score(sample_data, model, feature_means):
    """
    Calculate a health risk score based on the model's prediction and feature values.
    
    Args:
        sample_data (pandas.DataFrame): Data for the sample
        model: Trained ML model
        feature_means (pandas.Series): Mean values for the features
        
    Returns:
        float: Health risk score (0-100)
    """
    # Get the prediction probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(sample_data)[0][1]  # Probability of positive class
    else:
        prob = float(model.predict(sample_data)[0])  # Just use the prediction (0 or 1)
    
    # Calculate how far the sample is from the mean for key risk factors
    risk_factors = ['age', 'cholesterol', 'resting_bp', 'max_heart_rate']
    risk_factors = [rf for rf in risk_factors if rf in sample_data.columns]
    
    # If no risk factors are found, return just the probability score
    if not risk_factors:
        return prob * 100
    
    # Calculate deviations from mean for risk factors
    deviations = []
    for factor in risk_factors:
        if factor in sample_data.columns and factor in feature_means.index:
            # For max_heart_rate, lower is worse (inverted risk)
            if factor == 'max_heart_rate':
                if sample_data[factor].iloc[0] < feature_means[factor]:
                    deviation = (feature_means[factor] - sample_data[factor].iloc[0]) / feature_means[factor]
                    deviations.append(deviation)
            # For others, higher is worse
            else:
                if sample_data[factor].iloc[0] > feature_means[factor]:
                    deviation = (sample_data[factor].iloc[0] - feature_means[factor]) / feature_means[factor]
                    deviations.append(deviation)
    
    # Calculate average deviation (if we have any)
    if deviations:
        avg_deviation = sum(deviations) / len(deviations)
        # Combine probability and deviation for a composite score
        score = (0.7 * prob + 0.3 * min(avg_deviation, 1.0)) * 100
    else:
        score = prob * 100
    
    # Ensure score is between 0 and 100
    return min(max(score, 0), 100)

def get_risk_factors_explanation(sample_data, importance_df, threshold=0.05):
    """
    Generate an explanation of the main risk factors for a patient.
    
    Args:
        sample_data (pandas.DataFrame): Data for the sample
        importance_df (pandas.DataFrame): Feature importance data
        threshold (float): Minimum importance to include a feature
        
    Returns:
        str: Explanation of risk factors
    """
    # Filter to only include important features
    important_features = importance_df[importance_df['Importance'] > threshold]
    
    if len(important_features) == 0:
        return "No significant risk factors identified."
    
    # Get the top features
    top_features = important_features.head(5)['Feature'].tolist()
    
    # Create explanation
    explanation = "The main risk factors for this patient are:\n\n"
    
    for feature in top_features:
        if feature in sample_data.columns:
            value = sample_data[feature].iloc[0]
            formatted_feature = format_feature_name(feature)
            explanation += f"- **{formatted_feature}**: {value}\n"
    
    return explanation

def generate_recommendations(risk_score, risk_factors):
    """
    Generate health recommendations based on risk score and factors.
    
    Args:
        risk_score (float): Health risk score
        risk_factors (list): List of main risk factors
        
    Returns:
        str: Health recommendations
    """
    recommendations = "### Health Recommendations\n\n"
    
    # General recommendations based on risk score
    if risk_score < 20:
        recommendations += "Your risk level is **low**. Continue with regular health check-ups.\n\n"
    elif risk_score < 50:
        recommendations += "Your risk level is **moderate**. Consider lifestyle changes and regular monitoring.\n\n"
    else:
        recommendations += "Your risk level is **high**. Please consult a healthcare professional promptly.\n\n"
    
    # Specific recommendations based on risk factors
    recommendations += "Based on your risk factors:\n\n"
    
    common_factors = ['age', 'sex', 'cholesterol', 'resting_bp', 'max_heart_rate', 'exercise_angina', 'st_depression']
    
    for factor in risk_factors:
        if factor in common_factors:
            if factor == 'cholesterol':
                recommendations += "- **High Cholesterol**: Consider a diet low in saturated fats and high in fiber. Regular exercise can also help manage cholesterol levels.\n"
            elif factor == 'resting_bp':
                recommendations += "- **Blood Pressure**: Maintain a low-sodium diet, regular physical activity, and consider stress reduction techniques.\n"
            elif factor == 'max_heart_rate':
                recommendations += "- **Heart Rate**: Regular cardiovascular exercise can improve heart health. Consult a doctor before starting a new exercise program.\n"
            elif factor == 'exercise_angina':
                recommendations += "- **Exercise-induced Chest Pain**: This is a significant risk factor. Please consult a cardiologist for proper evaluation.\n"
            elif factor == 'st_depression':
                recommendations += "- **ST Depression**: This indicates potential heart issues. Medical evaluation is strongly recommended.\n"
    
    # General heart health recommendations
    recommendations += "\n**General Heart Health Tips:**\n"
    recommendations += "- Maintain a heart-healthy diet rich in fruits, vegetables, and whole grains\n"
    recommendations += "- Exercise regularly (aim for at least 150 minutes of moderate activity per week)\n"
    recommendations += "- Avoid smoking and limit alcohol consumption\n"
    recommendations += "- Manage stress through techniques like meditation or yoga\n"
    recommendations += "- Get adequate sleep (7-8 hours nightly)\n"
    
    return recommendations

def create_comparative_analysis(patient_data, population_data, features):
    """
    Create comparative analysis between a patient and the population.
    
    Args:
        patient_data (pandas.DataFrame): Data for the patient
        population_data (pandas.DataFrame): Data for the population
        features (list): Features to include in the analysis
        
    Returns:
        plotly.graph_objects.Figure: Comparative visualization
    """
    # Create the figure
    fig = go.Figure()
    
    # Format feature names
    formatted_features = [format_feature_name(feat) for feat in features]
    
    # Add patient data
    patient_values = [patient_data[feat].iloc[0] for feat in features]
    fig.add_trace(go.Bar(
        x=formatted_features,
        y=patient_values,
        name='Patient',
        marker_color='rgba(255, 65, 54, 0.7)'
    ))
    
    # Add population means
    population_means = [population_data[feat].mean() for feat in features]
    fig.add_trace(go.Bar(
        x=formatted_features,
        y=population_means,
        name='Population Average',
        marker_color='rgba(49, 130, 189, 0.7)'
    ))
    
    # Update layout
    fig.update_layout(
        title="Patient vs. Population Comparison",
        xaxis_title="Features",
        yaxis_title="Values",
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def get_confusion_matrix_explanation():
    """
    Return an explanation of the confusion matrix for educational purposes.
    
    Returns:
        str: Explanation of confusion matrix
    """
    explanation = """
    ### Understanding the Confusion Matrix
    
    The confusion matrix shows how well our model classifies patients:
    
    - **True Positives (TP)**: Correctly identified patients with heart disease
    - **True Negatives (TN)**: Correctly identified healthy patients
    - **False Positives (FP)**: Healthy patients incorrectly identified as having heart disease
    - **False Negatives (FN)**: Patients with heart disease incorrectly identified as healthy
    
    In healthcare:
    - False Negatives (missing a disease) are often more serious than False Positives
    - High sensitivity (recall) is important for screening tests
    - High specificity is important for confirmatory tests
    
    The ideal model would have high numbers along the diagonal (TP and TN) and low numbers off the diagonal (FP and FN).
    """
    return explanation

def get_metric_explanations():
    """
    Return explanations of common ML metrics for educational purposes.
    
    Returns:
        dict: Dictionary of metric explanations
    """
    explanations = {
        'accuracy': """
        **Accuracy**: The proportion of all predictions that were correct.
        - Formula: (TP + TN) / (TP + TN + FP + FN)
        - In healthcare: Simple but can be misleading if classes are imbalanced
        """,
        
        'precision': """
        **Precision**: The proportion of positive identifications that were actually correct.
        - Formula: TP / (TP + FP)
        - In healthcare: High precision means when we predict disease, we're usually right
        """,
        
        'recall': """
        **Recall (Sensitivity)**: The proportion of actual positives that were identified correctly.
        - Formula: TP / (TP + FN)
        - In healthcare: High recall means we rarely miss patients with the disease
        """,
        
        'f1': """
        **F1 Score**: The harmonic mean of precision and recall.
        - Formula: 2 * (Precision * Recall) / (Precision + Recall)
        - In healthcare: Balances the trade-off between precision and recall
        """,
        
        'specificity': """
        **Specificity**: The proportion of actual negatives that were identified correctly.
        - Formula: TN / (TN + FP)
        - In healthcare: High specificity means we rarely misdiagnose healthy patients
        """,
        
        'roc_auc': """
        **ROC AUC**: Area Under the Receiver Operating Characteristic Curve.
        - Measures how well the model can distinguish between classes across thresholds
        - Values range from 0.5 (no better than random) to 1.0 (perfect classification)
        - In healthcare: Used to evaluate how well a model can separate disease from non-disease
        """
    }
    
    return explanations

def get_ml_concept_explanations():
    """
    Return explanations of ML concepts for educational purposes.
    
    Returns:
        dict: Dictionary of concept explanations
    """
    explanations = {
        'train_test_split': """
        **Train-Test Split**: Dividing data into separate sets for training and evaluation.
        
        - **Why it's important**: To evaluate the model on data it hasn't seen during training
        - **Train set**: Used to train the model and learn patterns
        - **Test set**: Used to evaluate how well the model generalizes
        - **Common split**: 80% train, 20% test (or 70/30)
        
        Without this split, we risk overfitting and wouldn't know how the model performs on new data.
        """,
        
        'feature_scaling': """
        **Feature Scaling**: Transforming features to a similar scale.
        
        - **Why it's important**: Many algorithms are sensitive to feature magnitudes
        - **StandardScaler**: Mean=0, Standard deviation=1
        - **MinMaxScaler**: Scales features to a range, usually [0,1]
        
        Improves convergence for algorithms like gradient descent and prevents features with larger values from dominating the model.
        """,
        
        'overfitting': """
        **Overfitting**: When a model learns the training data too well, including the noise.
        
        - **Signs**: High training accuracy but low test accuracy
        - **Causes**: Too complex model, too few training examples
        - **Solutions**: More data, simpler model, regularization, cross-validation
        
        An overfitted model memorizes rather than generalizes, performing poorly on new data.
        """,
        
        'underfitting': """
        **Underfitting**: When a model is too simple to capture the underlying pattern.
        
        - **Signs**: Low accuracy on both training and test data
        - **Causes**: Too simple model, not enough features
        - **Solutions**: More complex model, better features, less regularization
        
        An underfitted model fails to capture important relationships in the data.
        """,
        
        'feature_engineering': """
        **Feature Engineering**: The process of creating, transforming, or selecting features.
        
        - **Why it's important**: Good features can dramatically improve model performance
        - **Types**: Feature creation, transformation, selection
        - **Examples**: Creating interaction terms, polynomial features, binning
        
        Often requires domain knowledge to create meaningful features that help the model learn relevant patterns.
        """,
        
        'cross_validation': """
        **Cross-Validation**: Technique for evaluating models by testing on multiple data splits.
        
        - **Why it's important**: More robust evaluation than a single train-test split
        - **k-fold CV**: Split data into k folds, train on k-1 and test on the remaining fold, rotate k times
        - **Benefits**: Uses all data for both training and testing, reduces variance in evaluation
        
        Helps ensure model performance isn't dependent on a specific data split.
        """
    }
    
    return explanations

def display_learning_resources():
    """
    Display additional learning resources for beginners.
    
    Returns:
        str: HTML formatted string with learning resources
    """
    resources = """
    ### Additional Learning Resources
    
    #### Online Courses
    - [Coursera: Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)
    - [edX: Data Science Essentials](https://www.edx.org/course/data-science-essentials)
    - [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
    
    #### Books
    - "Python Machine Learning" by Sebastian Raschka
    - "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurélien Géron
    - "The Hundred-Page Machine Learning Book" by Andriy Burkov
    
    #### Healthcare ML Resources
    - [MIT Clinical Machine Learning Group](https://clinicalml.org/)
    - [IEEE Journal of Biomedical and Health Informatics](https://www.embs.org/jbhi/)
    - [Stanford Medicine Big Data in Biomedicine](https://med.stanford.edu/bigdata.html)
    
    #### Python Libraries Documentation
    - [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
    - [Pandas](https://pandas.pydata.org/docs/)
    - [Matplotlib](https://matplotlib.org/stable/contents.html)
    - [Plotly](https://plotly.com/python/)
    """
    return resources

def create_ml_workflow_diagram():
    """
    Create a diagram representation of the ML workflow.
    
    Returns:
        str: HTML/SVG string with a diagram of the ML workflow
    """
    # Using SVG to create a simple workflow diagram
    svg = """
    <svg width="700" height="300" xmlns="http://www.w3.org/2000/svg">
        <!-- Define workflow steps -->
        <rect x="10" y="100" width="120" height="60" rx="10" ry="10" fill="#3498db" />
        <text x="70" y="130" font-family="Arial" font-size="12" fill="white" text-anchor="middle">Problem Definition</text>
        
        <rect x="160" y="100" width="120" height="60" rx="10" ry="10" fill="#3498db" />
        <text x="220" y="130" font-family="Arial" font-size="12" fill="white" text-anchor="middle">Data Collection</text>
        
        <rect x="310" y="100" width="120" height="60" rx="10" ry="10" fill="#3498db" />
        <text x="370" y="130" font-family="Arial" font-size="12" fill="white" text-anchor="middle">Data Exploration</text>
        
        <rect x="460" y="100" width="120" height="60" rx="10" ry="10" fill="#3498db" />
        <text x="520" y="130" font-family="Arial" font-size="12" fill="white" text-anchor="middle">Data Preprocessing</text>
        
        <rect x="160" y="190" width="120" height="60" rx="10" ry="10" fill="#3498db" />
        <text x="220" y="220" font-family="Arial" font-size="12" fill="white" text-anchor="middle">Model Evaluation</text>
        
        <rect x="310" y="190" width="120" height="60" rx="10" ry="10" fill="#3498db" />
        <text x="370" y="220" font-family="Arial" font-size="12" fill="white" text-anchor="middle">Model Training</text>
        
        <rect x="460" y="190" width="120" height="60" rx="10" ry="10" fill="#3498db" />
        <text x="520" y="220" font-family="Arial" font-size="12" fill="white" text-anchor="middle">Feature Engineering</text>
        
        <!-- Define arrows -->
        <svg>
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#2c3e50" />
                </marker>
            </defs>
        </svg>
        
        <!-- Draw arrows -->
        <line x1="130" y1="130" x2="160" y2="130" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrowhead)" />
        <line x1="280" y1="130" x2="310" y2="130" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrowhead)" />
        <line x1="430" y1="130" x2="460" y2="130" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrowhead)" />
        <line x1="520" y1="160" x2="520" y2="190" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrowhead)" />
        <line x1="460" y1="220" x2="430" y2="220" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrowhead)" />
        <line x1="310" y1="220" x2="280" y2="220" stroke="#2c3e50" stroke-width="2" marker-end="url(#arrowhead)" />
    </svg>
    """
    return svg

def get_healthcare_ml_examples():
    """
    Return examples of ML applications in healthcare.
    
    Returns:
        str: Information about healthcare ML applications
    """
    examples = """
    ### Real-world Applications of ML in Healthcare
    
    #### Diagnostic Tools
    - **Medical Imaging Analysis**: ML algorithms can detect patterns in X-rays, MRIs, and CT scans to assist in diagnosis of conditions like cancer, fractures, and neurological disorders.
    - **Pathology**: ML helps analyze tissue samples and blood tests to identify diseases and abnormalities.
    
    #### Predictive Analytics
    - **Risk Prediction**: Similar to our heart disease prediction, models can predict risk for conditions like diabetes, stroke, and readmission.
    - **Disease Progression**: Models can forecast how conditions like Alzheimer's or Parkinson's might progress in individual patients.
    
    #### Treatment Optimization
    - **Personalized Medicine**: ML helps tailor treatments based on patient characteristics and predicted responses.
    - **Drug Discovery**: Accelerates the identification of potential drug candidates by analyzing molecular structures.
    
    #### Operational Improvements
    - **Hospital Resource Management**: Predicting patient admission rates to optimize staffing and resource allocation.
    - **Fraud Detection**: Identifying unusual billing patterns that may indicate healthcare fraud.
    
    #### Remote Monitoring
    - **Wearable Devices**: Analyzing data from wearables to detect abnormal patterns that may indicate health issues.
    - **Telemedicine**: ML-powered diagnostic tools that can be used remotely.
    """
    return examples

def assess_data_quality(data):
    """
    Assess the quality of a dataset for machine learning.
    
    Args:
        data (pandas.DataFrame): The dataset to assess
        
    Returns:
        dict: Dictionary with data quality metrics
    """
    quality = {}
    
    # Check for completeness
    null_counts = data.isnull().sum()
    quality['missing_values'] = null_counts.to_dict()
    quality['completeness'] = 1 - (null_counts.sum() / (data.shape[0] * data.shape[1]))
    
    # Check for duplicates
    duplicate_rows = data.duplicated().sum()
    quality['duplicate_rows'] = duplicate_rows
    quality['duplicate_percentage'] = duplicate_rows / data.shape[0] if data.shape[0] > 0 else 0
    
    # Check for class balance (assuming 'target' column exists)
    if 'target' in data.columns:
        target_counts = data['target'].value_counts()
        quality['class_counts'] = target_counts.to_dict()
        
        # Calculate class imbalance ratio
        if len(target_counts) > 1:
            majority = target_counts.max()
            minority = target_counts.min()
            quality['imbalance_ratio'] = majority / minority
        else:
            quality['imbalance_ratio'] = float('inf')  # Only one class present
    
    # Feature type assessment
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    quality['numeric_features'] = len(numeric_features)
    quality['categorical_features'] = len(categorical_features)
    
    # Value range assessment for numeric features
    if numeric_features:
        quality['value_ranges'] = {col: {'min': data[col].min(), 'max': data[col].max()} for col in numeric_features}
    
    return quality
