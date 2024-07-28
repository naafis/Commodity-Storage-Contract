# lIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# Load raw data from csv
loaners = pd.read_csv('Task_3_and_4_Loan_Data.csv')
df = loaners.copy()
df.head()

df.info()
df.shape

df.describe()

df.drop(columns=['customer_id'], inplace=True)

# Ratios
df['loan_to_debt'] = df['loan_amt_outstanding'] / df['total_debt_outstanding']
df['loan_to_income'] = df['loan_amt_outstanding'] / df['income']
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']
# Interaction terms
df['loan_empyears_interac'] = df['loan_amt_outstanding'] * df['years_employed']
df['debt_empyears_interac'] = df['total_debt_outstanding'] * df['years_employed']
df['income_fico_interac'] = df['income'] * df['fico_score']
df['crlines_fico_interac'] = df['credit_lines_outstanding'] * df['fico_score']
df['crlines_income_interac'] = df['credit_lines_outstanding'] * df['income']
df['crlines_empyears_interac'] = df['credit_lines_outstanding'] * df['years_employed']
df['empyears_fico_interac'] = df['years_employed'] * df['fico_score']

# Extract feature skewness
skewness = df.skew()
print(skewness)

# Apply log transformation on positively skewed features
df['credit_lines_outstanding'] = np.log1p(df['credit_lines_outstanding'])
df['total_debt_outstanding'] = np.log1p(df['total_debt_outstanding'])
df['loan_to_debt'] = np.log1p(df['loan_to_debt'])
df['debt_empyears_interac'] = np.log1p(df['debt_empyears_interac'])
df['crlines_fico_interac'] = np.log1p(df['crlines_fico_interac'])
df['crlines_income_interac'] = np.log1p(df['crlines_income_interac'])
df['crlines_empyears_interac'] = np.log1p(df['crlines_empyears_interac'])

# Separate target and features
features = df.drop(columns=['default'])
target = df['default']

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

features_scaled_df.head()

# Find the correlation matrix
corr_matrix = features_scaled_df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Define threshold for feature selection
threshold = 0.85

# Extract highly correlated pairs
features_used = set()
features_removed = set()

for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[1]):
        if corr_matrix.iloc[i, j] >= threshold or corr_matrix.iloc[i, j] <= -threshold:
            left_feature = corr_matrix.index[i]
            right_feature = corr_matrix.columns[j]

            if right_feature not in features_used:
                features_used.add(right_feature)
                features_removed.add(left_feature)

# Convert sets to lists for print
features_used = list(features_used)
features_removed = list(features_removed)

print('Features to be used:', features_used)
print('Features removed:', features_removed)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled_df, target, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Extract feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features_scaled_df.columns,
                                      'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df

# Select features with high importance
main_features = ['debt_to_income', 'loan_to_debt', 'crlines_income_interac',
                 'empyears_fico_interac', 'fico_score']

# Calculate VIF for each feature
X = features_scaled_df[main_features]
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

# Apply SMOTE to balance train set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Retrain the random forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print('ROC AUC score:', roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Fit the model on the balanced training data
xgb_model.fit(X_train_balanced, y_train_balanced)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print("ROC AUC score:", roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Predict probabilities for calculating the probability of default
xgb_pd = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities of the positive class
expected_loss = xgb_pd * (1 - 0.1)

# Integrate cross-validation to test robustness
cross_val_scores = cross_val_score(xgb_model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
print("Cross-Validation AUC scores:", cross_val_scores)
print("Mean AUC score:", cross_val_scores.mean())


# Define function to plot learning curves
def plot_learning_curves(estimator, X, y, scoring='accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring=scoring, n_jobs=1,
        train_sizes=np.linspace(0.1, 1, 10), random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel(scoring.capitalize())
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')
    plt.legend(loc='best')
    plt.show()

# PLot learning curves for XGBoost
plot_learning_curves(xgb_model, X_train_balanced, y_train_balanced, scoring='roc_auc')