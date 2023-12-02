from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
df = pd.read_csv('LongCovidData.csv')

# Drop the index column
df = df.drop(columns=['Unnamed: 0'])

# Recoding LongCovid to Binary Representation
df['LongCovid'] = df['LongCovid'].apply(lambda x: 0 if x == 2 else 1)

# Define your features and target variable
X = df.drop('LongCovid', axis=1)
y = df['LongCovid']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Decision Tree Classifier
dt = DecisionTreeClassifier(criterion='gini', random_state=42)

# Train the model
dt.fit(X_train_scaled, y_train)

# Make predictions
y_pred_dt = dt.predict(X_test_scaled)

# Evaluate the model
report_dt = classification_report(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Print the evaluation results
print(report_dt)
print(conf_matrix_dt)

# Calculate and print AUROC and AUPRC
# Note: We use the predicted probabilities of the positive class
y_pred_proba_dt = dt.predict_proba(X_test_scaled)[:, 1]
roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
precision_recall_auc_dt = average_precision_score(y_test, y_pred_proba_dt)

print("AUROC (Decision Tree): ", roc_auc_dt)
print("AUPRC (Decision Tree): ", precision_recall_auc_dt)

# Feature Selection and Hyperparameter Tuning (TO DO)
# Consider tuning hyperparameters like max_depth, min_samples_split, min_samples_leaf, etc.
# Feature selection can be explored as well for model optimization
