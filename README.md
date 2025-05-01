# 🚀 ML Explorer

A powerful web application for training, analyzing, and deploying machine learning models with an intuitive user interface.

## 🌟 Features

### 📊 Dataset Management

-   Upload and store preprocessed datasets
-   Load sample datasets (Titanic, Iris, Mushrooms)
-   Persistent storage for future use
-   Easy dataset selection and management

### 🎯 Model Training

-   Support for multiple ML algorithms:
    -   Logistic Regression
    -   Random Forest
    -   Support Vector Machine (SVM)
    -   XGBoost
-   Customizable model parameters
-   Automatic model saving and versioning
-   Performance metrics visualization

### 🔮 Predictions

-   Make predictions on new data
-   Download prediction results
-   Batch prediction support
-   Feature validation

### 📈 Visualization & Analysis

-   Comprehensive model performance metrics
-   SHAP values for feature importance
-   Interactive visualizations:
    -   Feature distributions
    -   Correlation matrices
    -   ROC curves
    -   Confusion matrices
-   Statistical analysis
-   Data quality monitoring

## 🛠️ Implementation

### Tech Stack

-   **Frontend**: Streamlit
-   **ML Libraries**: scikit-learn, XGBoost
-   **Data Processing**: pandas, numpy
-   **Visualization**: plotly, matplotlib, seaborn
-   **Model Analysis**: SHAP, Evidently AI

### Project Structure

```
├── app.py                 # Main application entry point
├── utils.py               # Utility functions for ML operations
├── pages/
│   ├── Home.py            # Landing page
│   ├── Dataset_Load.py    # Dataset management
│   ├── Train_Models.py    # Model training interface
│   ├── Upload_Predict.py  # Prediction interface
│   └── Visualization.py   # Model analysis and visualization
├── models/                # Directory for saved models
└── datasets/              # Directory for datasets
```

### Key Components

#### Model Training (`utils.py`)

-   Implements model training and saving
-   Handles feature importance extraction
-   Manages model persistence

#### Dataset Management (`Dataset_Load.py`)

-   Handles file uploads
-   Manages sample datasets
-   Provides dataset preview and information

#### Model Training Interface (`Train_Models.py`)

-   Model selection and parameter tuning
-   Training progress visualization
-   Performance metrics display

#### Prediction Interface (`Upload_Predict.py`)

-   New data upload and validation
-   Prediction generation
-   Results download

#### Visualization (`Visualization.py`)

-   Model performance analysis
-   Feature importance visualization
-   Data quality monitoring

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/Aditya5757raj/Healthcare-ML-Tutorial.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## 📝 Usage Guide

1. **Load Data**

    - Upload your dataset or select a sample dataset
    - Preview and validate your data

2. **Train Model**

    - Select your target variable
    - Choose a model type
    - Adjust model parameters
    - Train and evaluate the model

3. **Make Predictions**

    - Upload new data
    - Generate predictions
    - Download results

4. **Analyze Results**
    - View model performance metrics
    - Explore feature importance
    - Analyze data quality
    - Generate visualizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

-   Built with ❤️ using Streamlit
-   Sample datasets from various sources
-   ML libraries and tools from the Python ecosystem
