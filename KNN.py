import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_auc_score, average_precision_score

# Load the dataset
df = pd.read_csv('LongCovidData.csv')

# Preprocess the data (handling missing values, encoding, etc.)
# Dropping the index column
df = df.drop(columns=['Unnamed: 0'])
# Recoding LongCovid to Binary Representation 2 = false 1 = true
df['LongCovid'] = df['LongCovid'].apply(lambda x: 0 if x == 2 else 1)

# Define your features and target variable
X = df.drop('LongCovid', axis=1)  
y = df['LongCovid']       

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN model
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

# best_auroc = 0
# best_auprc = 0
# best_k_auroc = 0
# best_k_auprc = 0

# for k in range(1, 26):  # Test a range of k values
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
    
#     # Getting the probabilities for positive class
#     y_proba = knn.predict_proba(X_test)[:, 1]

#     # Calculating AUROC and AUPRC
#     auroc = roc_auc_score(y_test, y_proba)
#     auprc = average_precision_score(y_test, y_proba)

#     if auroc > best_auroc:
#         best_auroc = auroc
#         best_k_auroc = k

#     if auprc > best_auprc:
#         best_auprc = auprc
#         best_k_auprc = k

# print(f"Best K Value for AUROC: {best_k_auroc} with AUROC: {best_auroc}")
# print(f"Best K Value for AUPRC: {best_k_auprc} with AUPRC: {best_auprc}")

# -----------------------------------------------------------------------
# Best K Value for AUROC: 25 with AUROC: 0.6261863206648385
# Best K Value for AUPRC: 25 with AUPRC: 0.3166988682880273

# For k = 5 AUROC:  0.5753610785795892
# for k = 5 AUPRC:  0.2656687275344795

# The increase in time complexity probably isnt a good tradeoff for the increase in AUROC and AUPRC accuracy





