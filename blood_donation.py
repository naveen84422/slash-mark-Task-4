import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load Dataset
data = pd.read_csv('blood_donation.csv')  # Replace with actual dataset path

# Display first few rows
data.head()

# Check for missing values
data.fillna(data.mean(), inplace=True)

# Summary Statistics
print("Dataset Summary:\n", data.describe())

# Visualizing Data Distribution
sns.pairplot(data, hue='Donation')
plt.show()

# Feature Selection
features = ['Recency', 'Frequency', 'Monetary', 'Time']  # Adjust based on dataset
target = 'Donation'
X = data[features]
y = data[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Machine Learning Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

# Model Evaluation
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f'\n{model_name} Accuracy: {accuracy * 100:.2f}%')
    print(f'{model_name} Classification Report:\n', classification_report(y_true, y_pred))
    print(f'{model_name} Confusion Matrix:\n', confusion_matrix(y_true, y_pred))

# Evaluate Both Models
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("Gradient Boosting", y_test, y_pred_gb)

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters for Random Forest:", grid_search.best_params_)

# Final Model with Best Params
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)
evaluate_model("Optimized Random Forest", y_test, y_pred_best)

# Feature Importance
feature_importances = best_rf_model.feature_importances_
sns.barplot(x=features, y=feature_importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Blood Donation Prediction')
plt.show()

# Save Model
joblib.dump(best_rf_model, 'blood_donation_model.pkl')
print("Model saved as blood_donation_model.pkl")

# Load and Test Saved Model
loaded_model = joblib.load('blood_donation_model.pkl')
y_pred_loaded = loaded_model.predict(X_test)
evaluate_model("Loaded Model", y_test, y_pred_loaded)
