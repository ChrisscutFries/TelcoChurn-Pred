import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from time import perf_counter
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('telecom_customer_churn.csv')
data = data.rename(lambda x: x.lower().strip().replace(' ', '_'), axis='columns')

# Transforming skewed features
numeric_data = data.select_dtypes(include=['number'])
skewness = numeric_data.apply(skew)
print("Skewness of numerical features:", skewness)

# Define columns to transform
columns_to_transform = ['number_of_dependents', 'number_of_referrals', 'total_refunds', 
                        'total_extra_data_charges', 'total_long_distance_charges']

# Apply transformations
for column in columns_to_transform:
    data[f'{column}_sqrt'] = np.sqrt(data[column])
    data[f'{column}_log'] = np.log1p(data[column])

# Visualize skewness before and after transformations
skewness_before = data[columns_to_transform].apply(skew)
skewness_after_sqrt = data[[f'{column}_sqrt' for column in columns_to_transform]].apply(skew)
skewness_after_log = data[[f'{column}_log' for column in columns_to_transform]].apply(skew)

print("Skewness before transformation:", skewness_before)
print("Skewness after square root transformation:", skewness_after_sqrt)
print("Skewness after log transformation:", skewness_after_log)

# Visualize distributions before and after transformation
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))
for i, column in enumerate(columns_to_transform):
    sns.histplot(data[column], kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {column} (Before)')
    sns.histplot(data[f'{column}_sqrt'], kde=True, ax=axes[i, 1])
    axes[i, 1].set_title(f'Distribution of {column} (sqrt)')
    sns.histplot(data[f'{column}_log'], kde=True, ax=axes[i, 2])
    axes[i, 2].set_title(f'Distribution of {column} (log)')
    axes[i, 0].grid(True)
    axes[i, 1].grid(True)
    axes[i, 2].grid(True)

plt.tight_layout()
plt.show()

# Drop redundant columns
data = data.drop(['number_of_dependents', 'number_of_dependents_log', 'number_of_referrals', 'number_of_referrals_log',
                  'total_refunds', 'total_refunds_sqrt', 'total_extra_data_charges', 'total_extra_data_charges_sqrt',
                  'total_long_distance_charges', 'total_long_distance_charges_log'], axis=1)

# Visualizing missing values in object columns
plt.figure(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include=['object']).isnull(), cbar=False, cmap='viridis')
plt.title("Missing values in object columns")
plt.show()

# Visualizing missing values in numerical columns
plt.figure(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include=['number']).isnull(), cbar=False, cmap='viridis')
plt.title("Missing values in numerical columns")
plt.show()

# Fill missing values for specific columns
data['internet_type'] = data['internet_type'].apply(lambda x: 'no_internet_service' if pd.isnull(x) else x)
data['offer'] = data['offer'].apply(lambda x: 'no_offer' if pd.isnull(x) else x)

columns_to_fill = ['online_security', 'online_backup', 'device_protection_plan', 'premium_tech_support',
                   'streaming_tv', 'streaming_movies', 'streaming_music', 'unlimited_data']
for column in columns_to_fill:
    data[column] = data[column].apply(lambda x: 'no_internet_service' if pd.isnull(x) else x)

# Fill missing values for numerical columns
num_columns_to_fill = ['avg_monthly_long_distance_charges', 'avg_monthly_gb_download']
for column in num_columns_to_fill:
    data[column] = data[column].apply(lambda x: 0 if pd.isnull(x) else x)

# Encode categorical features
conversion_dict = {"Yes": 1, "No": 0}
data['multiple_lines'] = data['multiple_lines'].map(conversion_dict)
label_encoder = LabelEncoder()
data['customer_status'] = label_encoder.fit_transform(data['customer_status'])

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Imputation using KNNImputer
label_column = data['customer_status']
data = data.drop(columns=['customer_status'])
imputer = KNNImputer(n_neighbors=3)
imputed_data = imputer.fit_transform(data)
data = pd.DataFrame(data=imputed_data, columns=data.columns)
data['multiple_lines'] = data['multiple_lines'].apply(lambda x: round(x))
data['customer_status'] = label_column

# Split data
X = data.drop(['customer_status'], axis=1)
y = data['customer_status'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle outliers
def handle_outliers(X_train, X_test, columns):
    for col in columns:
        Q1 = np.percentile(X_train[col], 25)
        Q3 = np.percentile(X_train[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_train[col] = np.where(X_train[col] < lower_bound, lower_bound, X_train[col])
        X_train[col] = np.where(X_train[col] > upper_bound, upper_bound, X_train[col])
        X_test[col] = np.where(X_test[col] < lower_bound, lower_bound, X_test[col])
        X_test[col] = np.where(X_test[col] > upper_bound, upper_bound, X_test[col])
    return X_train, X_test

columns_to_handle = ['avg_monthly_gb_download', 'total_revenue', 'number_of_dependents_sqrt',
                     'total_refunds_log', 'total_extra_data_charges_log']
X_train, X_test = handle_outliers(X_train, X_test, columns_to_handle)

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTETomeks
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# Hyperparameter tuning and model training
xgb_model = xgb.XGBClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
start_time = perf_counter()
grid_search.fit(X_train_resampled, y_train_resampled)
end_time = perf_counter()

# Model evaluation
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Time taken for hyperparameter tuning: {end_time - start_time:.2f} seconds")
y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))