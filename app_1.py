import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
import joblib

# Generate sample data
def load_data():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    return X, y

# Train models
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

# Predict and evaluate
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for ROC
        
        accuracy = model.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_prob)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'ROC AUC': roc_auc,
            'Confusion Matrix': conf_matrix,
            'Classification Report': class_report
        }
    return results

# Streamlit app
def main():
    st.title("Model Performance and Predictions")

    # Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = train_models(X_train_scaled, y_train)
    
    # Model selection
    st.sidebar.header("Model Selection")
    model_names = list(models.keys())
    selected_model_name = st.sidebar.selectbox("Select Model", model_names)
    selected_model = models[selected_model_name]

    # Evaluate selected model
    results = evaluate_models(models, X_test_scaled, y_test)
    metrics = results[selected_model_name]

    st.header(f"Model: {selected_model_name}")
    
    st.subheader("Metrics")
    st.write(f"**Accuracy:** {metrics['Accuracy']:.4f}")
    st.write(f"**ROC AUC:** {metrics['ROC AUC']:.4f}")
    
    st.write("**Confusion Matrix:**")
    st.write(metrics['Confusion Matrix'])
    
    st.write("**Classification Report:**")
    st.text(metrics['Classification Report'])
    
    st.subheader("ROC Curve")
    plt.figure(figsize=(10, 7))
    y_prob = selected_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{selected_model_name} (AUC = {metrics["ROC AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    st.subheader("Feature Importance")
    if selected_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importances = selected_model.feature_importances_
        feature_names = [f'Feature {i}' for i in range(X_test_scaled.shape[1])]
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        ax.set_title(f'Feature Importance - {selected_model_name}')
        st.pyplot(fig)

    st.subheader("Make Predictions")
    input_data = st.text_input("Enter features separated by commas (e.g., 0.1, 0.2, ..., 0.5)")
    if input_data:
        try:
            # Convert input data to numpy array and reshape
            input_features = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)
            
            # Check if the number of features matches the model's input
            if input_features.shape[1] != X_train_scaled.shape[1]:
                st.error(f"Number of features should be {X_train_scaled.shape[1]}.")
            else:
                # Transform input features using the same scaler
                input_features_scaled = scaler.transform(input_features)
                
                # Predict using the selected model
                prediction = selected_model.predict(input_features_scaled)
                prediction_proba = selected_model.predict_proba(input_features_scaled)[:, 1]
                st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
                st.write(f"Probability of Positive: {prediction_proba[0]:.4f}")
                
        except ValueError:
            st.error("Please enter valid numerical values separated by commas.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()