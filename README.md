# Heart Disease Prediction Project

## Overview

This project focuses on developing a machine learning model to predict the likelihood of heart disease based on various patient health parameters. It involves a comprehensive data science pipeline, from exploratory data analysis and rigorous preprocessing to model training, evaluation, and deployment as an interactive Streamlit application.

---

## Dataset

The project utilizes the `heart.csv` dataset, which contains medical records with features relevant to heart health. Key features include:

- **Age**: Age of the patient
- **Sex**: Gender (Male/Female)
- **ChestPainType**: Type of chest pain experienced
- **RestingBP**: Resting blood pressure
- **Cholesterol**: Serum cholesterol
- **FastingBS**: Fasting blood sugar
- **RestingECG**: Resting electrocardiogram results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: The slope of the peak exercise ST segment
- **HeartDisease**: Target variable (0 = No Heart Disease, 1 = Heart Disease)

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

- Initial inspection of data shape, information, and descriptive statistics
- Checked for duplicate records and missing values
- Visualized distributions of numerical features using histograms and categorical feature counts using bar plots
- Analyzed relationships between features and the target variable using count plots, box plots, violin plots, and correlation heatmaps

### 2. Data Cleaning and Preprocessing

- Handled missing/zero values in Cholesterol and RestingBP by imputing them with their respective means from non-zero values
- Converted categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) into numerical format using one-hot encoding (`pd.get_dummies` with `drop_first=True`)
- Scaled numerical features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) using `StandardScaler` to normalize their ranges

### 3. Feature Engineering and Selection

- Utilized Pearson correlation to assess linear relationships between numerical features and the target variable
- Applied Chi-squared tests to determine the independence and significance of categorical features with respect to the target variable, aiding in feature selection

### 4. Model Training and Evaluation

- The dataset was split into training and testing sets (80% train, 20% test)
- Multiple classification algorithms were trained and evaluated:
  - Logistic Regression
  - Gaussian Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Support Vector Machine (SVM)
- The K-Nearest Neighbors (KNN) classifier was identified as the best-performing model based on accuracy
- Model performance was evaluated using `accuracy_score`, `confusion_matrix`, and `classification_report`

### 5. Model Persistence

- The trained KNN model (`KNN_heart.pkl`), the fitted StandardScaler (`scaler.pkl`), and the ordered list of feature columns (`columns.pkl`) were saved using `joblib` for later use in the Streamlit application

---

## How to Run the Streamlit Application

1. **Clone the Repository (or download files):**
   - Ensure you have `heart_disease_app.py`, `KNN_heart.pkl`, `scaler.pkl`, and `columns.pkl` in the same directory (ideally, part of a GitHub repository).

2. **Install Dependencies:**
   - Make sure you have Python installed. Then, install the required libraries:
     ```bash
     pip install streamlit pandas numpy scikit-learn
     ```

3. **Run the App:**
   - Open your terminal or command prompt, navigate to the directory where you saved the files, and execute:
     ```bash
     streamlit run heart_disease_app.py
     ```

4. **Access the App:**
   - Your web browser will automatically open to the Streamlit application (usually at [http://localhost:8501](http://localhost:8501)). You can then input patient data and get real-time predictions.

---

## Files in this Project

- `heart.py`: Original Jupyter Notebook/script containing the full data analysis, preprocessing, model training, and saving steps
- `heart_disease_app.py`: The Streamlit application code for the interactive demo
- `KNN_heart.pkl`: The saved pre-trained K-Nearest Neighbors model
- `scaler.pkl`: The saved fitted StandardScaler object
- `columns.pkl`: A list containing the ordered names of the features expected by the model
- `heart.csv`: The dataset used for training the model

---

## Future Enhancements

- Integrate advanced ensemble methods (e.g., Random Forest, Gradient Boosting, XGBoost) and hyperparameter tuning (`GridSearchCV`, `RandomizedSearchCV`) to potentially further improve model accuracy and robustness
- Implement a more comprehensive web interface with data visualization directly within the Streamlit app
- Explore deploying the model as a REST API for broader integration

---

## License

[Specify your license here, e.g., MIT License. Update as appropriate for your project.]
