import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import uniform

# Load the cleaned datasets
train_clean = pd.read_csv('train_clean.csv')
test_clean = pd.read_csv('test_clean.csv')

# Prepare the data
X = train_clean.drop(columns=['Response', 'id'])  # Features
y = train_clean['Response']  # Target

# Split 10% of the training data for hyperparameter tuning
X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.8, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', seed=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'max_depth': [3, 4, 5],
    'eta': uniform(0.2, 0.5),  # Uniform distribution for eta between 0.1 and 0.4
    'subsample': uniform(0.7, 1),  # Uniform distribution for subsample between 0.8 and 1.0
    'colsample_bytree': uniform(0.6, 1),  # Uniform distribution for colsample_bytree between 0.8 and 1.0
    'n_estimators': [200,1000],
    'gamma': uniform(0.2, 1),
    'alpha': uniform(0.2, 0.9),  # Uniform distribution for alpha between 0 and 1
    'lambda': uniform(0, 0.6)  # Uniform distribution for lambda between 0 and 1
}

# Perform hyperparameter tuning using RandomizedSearchCV on 10% of the data
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, scoring='roc_auc', cv=2, n_iter=80, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_tune, y_tune)

# Get the best parameters
best_params = random_search.best_params_

print(f'Best Parameters: {best_params}')

# Train the model with the best hyperparameters on the entire dataset
best_model = xgb.XGBClassifier(objective='binary:logistic', seed=42, **best_params)
best_model.fit(X, y)

# Evaluate the model on the validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_val = best_model.predict_proba(X_val)[:, 1]
y_pred_val_bin = (y_pred_val > 0.5).astype(int)

accuracy = accuracy_score(y_val, y_pred_val_bin)
auc = roc_auc_score(y_val, y_pred_val)

print(f'Validation Accuracy: {accuracy}')
print(f'Validation AUC: {auc}')

# Prepare the test data
X_test = test_clean.drop(columns=['id'])

# Make predictions on the test set
y_pred_test = best_model.predict_proba(X_test)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({
    'id': test_clean['id'],
    'Response': y_pred_test
})

# Save the submission file
submission.to_csv('submission.csv', index=False)