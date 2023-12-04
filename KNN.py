import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
# Load the dataset
df = pd.read_csv('LongCovidData.csv')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Basic evaluation with classification report and confusion matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Additional evaluation for imbalanced dataset
roc_auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])

# Calculate AUPRC
precision_recall_auc = average_precision_score(y_test, knn.predict_proba(X_test)[:, 1])

print("AUROC: ", roc_auc)
print("AUPRC: ", precision_recall_auc)


############### Code to find most acurrate K ###############
# Define the parameter grid for k
# param_grid = {'n_neighbors': range(1, 31)}

# # Create a KNN model
# knn = KNeighborsClassifier()

# # Define the scoring function
# scorer = make_scorer(roc_auc_score, needs_proba=True)

# # Use GridSearchCV to find the best parameter 
# grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=scorer)
# grid_search.fit(X_train, y_train)

# # Print the best parameter and score
# print("Best parameter: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# # Use the best parameter to create a new model
# knn_best = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
# knn_best.fit(X_train, y_train)

# # Make predictions with the best model
# predictions = knn_best.predict(X_test)

# # Evaluate the model
# print(classification_report(y_test, predictions))

# -----------------------------------------------------------------------
#Best parameter:  {'n_neighbors': 30} AUROC 0.6214561327643617

# For k = 5 AUROC:  0.5753610785795892
# The increase in time complexity probably isnt a good tradeoff for the increase in AUROC and AUPRC accuracy





