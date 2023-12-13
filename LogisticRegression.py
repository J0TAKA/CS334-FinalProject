from sklearn.linear_model import LogisticRegression
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
X = df.drop(['LongCovid','Race','Education','Smoker','ECig','Drinker','ExerciseLevel','HeartAttack','HeartDisease','Stroke','Cancer','KidneyDisease','Diabetes','Urban'], axis=1)
y = df['LongCovid']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model
logreg = LogisticRegression(random_state=42,C= 0.001, max_iter= 100, penalty= 'l2', solver= 'saga')

# Train the model
logreg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test_scaled)

# Evaluate the model
report_logreg = classification_report(y_test, y_pred_logreg)
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)

# Print the evaluation results
print(report_logreg)
print(conf_matrix_logreg)

# Calculate and print AUROC and AUPRC
roc_auc_logreg = roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:, 1])
precision_recall_auc_logreg = average_precision_score(y_test, logreg.predict_proba(X_test_scaled)[:, 1])

print("AUROC (Logistic Regression): ", roc_auc_logreg)
print("AUPRC (Logistic Regression): ", precision_recall_auc_logreg)

# Feature Selection and Hyperparameter Tuning (TO DO)
# Consider tuning hyperparameters like C, penalty, etc.
# Feature selection can be explored as well for model optimization
