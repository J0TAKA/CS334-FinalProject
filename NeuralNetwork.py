from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
df = pd.read_csv('LongCovidData.csv')

# Preprocess the data (handling missing values, encoding, etc.)
# Dropping the index column
df = df.drop(columns=['Unnamed: 0'])
# Recoding LongCovid to Binary Representation 2 = false 1 = true
df['LongCovid'] = df['LongCovid'].apply(lambda x: 0 if x == 2 else 1)

# Load the dataset
df = pd.read_csv('LongCovidData.csv')

# Drop the index column
df = df.drop(columns=['Unnamed: 0'])

# Recoding LongCovid to Binary Representation
df['LongCovid'] = df['LongCovid'].apply(lambda x: 0 if x == 2 else 1)
df = df.drop(columns=['Pregnant', "Covid","State"])
#X = df.drop('LongCovid', axis=1)
#y = df['LongCovid']
X = df.drop('LongCovid', axis=1)
y = df['LongCovid']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Neural Network model
mlp = MLPClassifier(random_state=42, activation= "relu", alpha = 0.01, batch_size = 'auto', learning_rate="adaptive", verbose = True, validation_fraction=0.1)

# Train the model
mlp.fit(X_train_scaled, y_train)

# Make predictions
y_pred_mlp = mlp.predict(X_test_scaled)

# Evaluate the model
report_mlp = classification_report(y_test, y_pred_mlp)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)

print(report_mlp)
print(conf_matrix_mlp)

# Calculate and print AUROC and AUPRC for MLP Classifier
roc_auc_mlp = roc_auc_score(y_test, mlp.predict_proba(X_test_scaled)[:, 1])
precision_recall_auc_mlp = average_precision_score(y_test, mlp.predict_proba(X_test_scaled)[:, 1])

print("AUROC (MLP Classifier): ", roc_auc_mlp)
print("AUPRC (MLP Classifier): ", precision_recall_auc_mlp)

# Additional evaluation for imbalanced dataset (TO DO)
# Consider evaluating the model performance on an imbalanced dataset.
# Feature Selection and Hyperparameter Tuning (TO DO)
# Consider tuning hyperparameters like number of layers, neurons per layer, learning rate, etc.
# Feature selection can be explored as well for model optimization
