from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test_scaled)

# Evaluate the model
report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Print the evaluation results
print(report_rf)
print(conf_matrix_rf)

#MISSING OPTIMAL FEATURE SELECTION AND HYPERPARAMETER TUNNING!
