from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#from Preprocessing import return_data
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.under_sampling import NearMiss, RandomUnderSampler
#X, y = return_data()

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

#nm = NearMiss(version = 1 , n_neighbors = 5)

#x_balanced,y_balanced= nm.fit_resample(X,y)
#rus = RandomUnderSampler()
#x_balanced, y_balanced = rus.fit_sample(X,y)
#print(x_balanced.shape, y_balanced.shape)

#param_grid = {
#    'n_estimators': [10, 50, 100],
#    'max_depth': [None, 10, 20],
#    'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4],
#    'max_features': ['sqrt', 'log2'],
#    'bootstrap': [True],
#}

#grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
#grid_search.fit(X_train, y_train)
#print("Best Parameters:", grid_search.best_params_)
# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42,n_estimators =100,max_depth=20,bootstrap = True, max_features = "log2",min_samples_leaf = 4, min_samples_split = 2 )

# Train the model
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test_scaled)

# Evaluate the model
report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)


feature_names = [f"feature {i}" for i in range(X.shape[1])]

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
outpath = 'Feature_Importance.png'
plt.savefig(outpath, dpi=600)


# Print the evaluation results
print(report_rf)
print(conf_matrix_rf)

# Calculate and print AUROC and AUPRC using the scaled test data
roc_auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
precision_recall_auc_rf = average_precision_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])

print("AUROC (Random Forest): ", roc_auc_rf)
print("AUPRC (Random Forest): ", precision_recall_auc_rf)

# Feature Selection and Hyperparameter Tuning (TO DO)
# Consider using techniques like RFE, SelectFromModel, or analyzing feature importances
# Use GridSearchCV or RandomizedSearchCV for systematic hyperparameter optimization
