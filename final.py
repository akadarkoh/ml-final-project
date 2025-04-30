import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import umap

# Load Breast Cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer

# Load dataset and create DataFrame
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data Preprocessing
# Standardize the data for SVM and other algorithms
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction - PCA and UMAP

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reducing to 2 components for visualization
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Apply UMAP for dimensionality reduction
umap_model = umap.UMAP(n_components=2)
X_train_umap = umap_model.fit_transform(X_train_scaled)
X_test_umap = umap_model.transform(X_test_scaled)

# Model Training - Random Forest and SVM

# Random Forest Classifier (Initial Model)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)

# Support Vector Machine Classifier (Initial Model)
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
svm_score = svm.score(X_test_scaled, y_test)

# Hyperparameter Optimization using GridSearchCV for Random Forest

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=1)

grid_search_rf.fit(X_train_scaled, y_train)

# Best Random Forest model after hyperparameter tuning
best_rf = grid_search_rf.best_estimator_
best_rf_score = best_rf.score(X_test_scaled, y_test)

# Hyperparameter Optimization using GridSearchCV for SVM

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search_svm = GridSearchCV(estimator=SVC(random_state=42),
                               param_grid=param_grid_svm, cv=5, n_jobs=-1, verbose=1)

grid_search_svm.fit(X_train_scaled, y_train)

# Best SVM model after hyperparameter tuning
best_svm = grid_search_svm.best_estimator_
best_svm_score = best_svm.score(X_test_scaled, y_test)

# Visualization

# Print the results of different models
print(f"Random Forest (Initial) Accuracy: {rf_score:.4f}")
print(f"SVM (Initial) Accuracy: {svm_score:.4f}")
print(f"Best Random Forest Accuracy (after tuning): {best_rf_score:.4f}")
print(f"Best SVM Accuracy (after tuning): {best_svm_score:.4f}")

# Plot PCA results
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', s=50, alpha=0.7)
plt.title('PCA - Training Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Diagnosis')
plt.show()

# Plot UMAP results
plt.figure(figsize=(8, 6))
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='coolwarm', s=50, alpha=0.7)
plt.title('UMAP - Training Data')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.colorbar(label='Diagnosis')
plt.show()

# Conclusion on how dimensionality reduction and hyperparameter optimization impacted the model performance.
