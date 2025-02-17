import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Add at the top of the file, after imports
os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # Force single CPU
os.environ["JOBLIB_TEMP_FOLDER"] = "./joblib_temp"  # Set explicit temp directory

# Load Dataset
df = pd.read_csv("diabetes.csv")

def preprocess_diabetes_data(df):
    """
    Preprocess diabetes dataset with domain-specific knowledge
    """
    # Create a copy of the dataframe
    processed_df = df.copy()
    
    # Handle missing values (0s) with medical domain knowledge
    zero_value_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_value_columns:
        processed_df.loc[processed_df[column] == 0, column] = np.nan
        processed_df[column] = processed_df[column].fillna(processed_df[column].median())
    
    # BMI categories (medical standards)
    processed_df['BMI_Category'] = pd.cut(
        processed_df['BMI'], 
        bins=[0, 18.5, 24.9, 29.9, float('inf')],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    
    # Glucose levels categories
    processed_df['Glucose_Category'] = pd.cut(
        processed_df['Glucose'],
        bins=[0, 70, 99, 126, float('inf')],
        labels=['Low', 'Normal', 'Prediabetes', 'Diabetes']
    )
    
    # Feature interactions
    processed_df['Glucose_BMI'] = processed_df['Glucose'] * processed_df['BMI']
    processed_df['Age_BMI'] = processed_df['Age'] * processed_df['BMI']
    processed_df['Glucose_Age'] = processed_df['Glucose'] * processed_df['Age']
    
    # Safe division for Insulin/Glucose ratio
    processed_df['Insulin_Glucose'] = np.where(
        processed_df['Glucose'] != 0,
        processed_df['Insulin'] / processed_df['Glucose'],
        0
    )
    
    # Convert categorical variables to dummy variables
    processed_df = pd.get_dummies(
        processed_df, 
        columns=['BMI_Category', 'Glucose_Category'],
        prefix=['BMI', 'Glucose']
    )
    
    return processed_df

# Main execution
def train_diabetes_models():
    # Preprocess the data
    print("Preprocessing data...")
    df_processed = preprocess_diabetes_data(df)
    
    # Prepare features and target
    X = df_processed.drop(columns=['Outcome'])
    y = df_processed['Outcome']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Optimized model configurations based on previous results
    param_grids = {
        "Random Forest": {
            'model': RandomForestClassifier(
                random_state=42,
                n_jobs=1,
                bootstrap=True,
                oob_score=True,
                warm_start=True,
                class_weight='balanced'  # Set this directly
            ),
            'params': {
                'model__n_estimators': [500, 700],
                'model__max_depth': [11, 13],  # Increase depth slightly
                'model__min_samples_split': [2],
                'model__min_samples_leaf': [1],
                'model__max_features': ['sqrt'],
                'model__criterion': ['gini']  # Gini performed better
            }
        },
        "Gradient Boosting": {
            'model': GradientBoostingClassifier(
                random_state=42,
                verbose=0,
                validation_fraction=0.1,
                warm_start=True  # Add warm start
            ),
            'params': {
                'model__n_estimators': [300, 400],
                'model__learning_rate': [0.08, 0.1],  # Fine-tune learning rate
                'model__max_depth': [5, 6],
                'model__subsample': [0.9, 1.0],
                'model__min_samples_split': [4, 5],
                'model__max_features': ['sqrt'],
                'model__min_samples_leaf': [2, 3]  # Add leaf size control
            }
        },
        "Extra Trees": {  # Replace XGBoost with Extra Trees
            'model': ExtraTreesClassifier(
                random_state=42,
                n_jobs=1,
                bootstrap=True,
                oob_score=True,
                warm_start=True
            ),
            'params': {
                'model__n_estimators': [200, 300],
                'model__max_depth': [5, 7],
                'model__min_samples_split': [2, 3],
                'model__min_samples_leaf': [1, 2],
                'model__max_features': ['sqrt'],
                'model__class_weight': ['balanced'],
                'model__criterion': ['gini', 'entropy']
            }
        }
    }

    # Enhanced evaluation metrics with threshold optimization
    def evaluate_model(model, X_test, y_test, name):
        y_pred_proba = model.predict_proba(X_test)[:,1]
        
        # Find optimal threshold
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Use optimal threshold for predictions
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Calculate metrics
        specificity = classification_report(y_test, y_pred, output_dict=True)['0']['recall']
        sensitivity = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nDetailed Metrics for {name}:")
        print(f"Optimal Threshold: {best_threshold:.3f}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
        print(f"Specificity (True Negative Rate): {specificity:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'threshold': best_threshold,
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    # Train and evaluate models
    best_models = {}
    model_metrics = {}
    
    for name, config in param_grids.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', config['model'])
        ], memory='cachedir')
        
        grid_search = GridSearchCV(
            pipeline,
            config['params'],
            cv=5,
            scoring='roc_auc',
            n_jobs=1,
            verbose=1,
            error_score='raise'
        )
        
        try:
            grid_search.fit(X_train_balanced, y_train_balanced)
            best_models[name] = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Cross-validation score: {grid_search.best_score_:.4f}")
            
            # Evaluate model with enhanced metrics
            model_metrics[name] = evaluate_model(
                grid_search.best_estimator_,
                X_test,
                y_test,
                name
            )
            
            # Save model
            model_filename = f"{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(grid_search.best_estimator_, model_filename)
            print(f"Model saved as {model_filename}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue

    # Print comparative results
    print("\nComparative Model Performance:")
    for name, metrics in model_metrics.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Cleanup cache directory
    try:
        import shutil
        shutil.rmtree('cachedir', ignore_errors=True)
    except:
        pass

    # Save feature names with error handling
    try:
        joblib.dump(list(X.columns), "feature_names.pkl")
    except Exception as e:
        print(f"Error saving feature names: {str(e)}")

    # Feature importance visualization with error handling
    try:
        if hasattr(best_models["Random Forest"].named_steps['model'], "feature_importances_"):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_models["Random Forest"].named_steps['model'].feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
            plt.title('Top 15 Feature Importance (Random Forest)')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"Error creating feature importance plot: {str(e)}")
    
    return best_models

if __name__ == "__main__":
    try:
        best_models = train_diabetes_models()
        print("\nTraining complete. Models and preprocessors saved.")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
