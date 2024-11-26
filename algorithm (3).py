# Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# reading the dataset
file_path = r"C:\Users\ap9029\Desktop\Princeton\Data\PCOS Dataset.csv"
data = pd.read_csv(file_path)

data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')
data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')

data['Marraige Status (Yrs)'] = data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median())
data['II    beta-HCG(mIU/mL)'] = data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median())
data['AMH(ng/mL)'] = data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median())
data['Fast food (Y/N)'] = data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].mode()[0])

# Clearing up the extra space in the column names (optional)
data.columns = [col.strip() for col in data.columns]

# Identifying non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

# Converting non-numeric columns to numeric where possible
for col in non_numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Dropping rows with any remaining non-numeric values
data.dropna(inplace=True)

# Preparing data for model training
columns_to_drop = ["PCOS (Y/N)", "Sl. No", "Patient File No.", "Marraige Status (Yrs)", "Blood Group", "II    beta-HCG(mIU/mL)", "TSH (mIU/L)", "Waist:Hip Ratio"]
X = data.drop(columns=columns_to_drop)
y = data["PCOS (Y/N)"]

# Splitting the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fitting the RandomForestClassifier to the training set
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled, y_train)

# Making prediction and checking the test set
pred_rfc = rfc.predict(X_test_scaled)
accuracy = accuracy_score(y_test, pred_rfc)
print(accuracy)

# Cross-Validation
cross_val_scores = cross_val_score(rfc, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean cross-validation score: {cross_val_scores.mean()}")

# Hyperparameter Tuning using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(rfc, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
print(f"Best hyperparameters: {grid_search.best_params_}")

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, pred_rfc)
print(f"Confusion Matrix:\n{cm}")
print(f"Classification Report:\n{classification_report(y_test, pred_rfc)}")

# Feature Importance
feature_importance = rfc.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
print(f"Feature Importance:\n{importance_df.sort_values(by='Importance', ascending=False)}")

# Model Saving and Loading
joblib.dump(rfc, 'random_forest_model.pkl')  # Save the model
rfc_loaded = joblib.load('random_forest_model.pkl')  # Load the model

# Plotting Learning Curves
train_sizes, train_scores, test_scores = learning_curve(rfc, X_train_scaled, y_train, cv=5)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test score')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Example: Collecting user input for the features
print("Please enter the following details:")

user_input = []
provided_columns = []

# Loop over the columns of X
for column in X.columns:
    # Prompt the user to enter the value for each feature
    input_value = input(f"Enter the value for {column}: ")
    if len(input_value) > 0:
        value = float(input_value)
        # Append the input value to the user_input list
        user_input.append(value)
        # Keep track of columns that have been provided
        provided_columns.append(column)

# Create a DataFrame with the provided columns
user_input_df = pd.DataFrame([user_input], columns=provided_columns)

# Ensure the user input DataFrame has the same columns as the training data
for col in X.columns:
    if col not in user_input_df.columns:
        user_input_df[col] = X_train[col].median()

# Reorder columns to match the training data
user_input_df = user_input_df[X.columns]

# Scale the user input
user_input_scaled = scaler.transform(user_input_df)

# Get the probability of PCOS
probabilities = rfc.predict_proba(user_input_scaled)

# Extract probability for PCOS (class 1)
probability_pcos = probabilities[0][1]  # Probability of PCOS (class 1)
probability_non_pcos = probabilities[0][0]  # Probability of non-PCOS (class 0)

# Output the result
print(f"Probability of PCOS: {probability_pcos * 100:.2f}%")
print(f"Probability of non-PCOS: {probability_non_pcos * 100:.2f}%")
