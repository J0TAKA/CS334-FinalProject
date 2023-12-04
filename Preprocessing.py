import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import researchpy as rp

def main():
    # Load the dataset
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
    df = df.drop(columns=['Pregnant',"Covid"])


    sns.set()
    plt.figure(figsize=(20, 15))
    ax = plt.subplot(111)
    # Calculate the correlation matrix
    corrMat = df.corr(method='spearman')
    #cramersv_matrix = cramers_v(df)
    # Create the heatmap
    ax = sns.heatmap(corrMat, annot=True, annot_kws={"fontsize": 8})
    #ax = sns.heatmap(cramersv_matrix, annot=True, annot_kws={"fontsize": 8})
    # Export the heatmap
    outpath = 'Spearman_Train_heatmap.png'
    plt.savefig(outpath, dpi=600)
    # Define your features and target variable
    X = df.drop('LongCovid', axis=1)
    y = df['LongCovid']
    #return_data(df,y)

# Chi-Squared and Cramér's V
chi_squared_results = {}
cramers_v_results = {}

for col in categorical_cols:
    crosstab = pd.crosstab(df['LongCovid'], df[col])
    chi2, p, _, _ = chi2_contingency(crosstab)
    chi_squared_results[col] = p
    cramers_v_results[col] = cramers_v(df['LongCovid'], df[col])

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, columns=categorical_cols)

# Splitting data into features and target
X = df.drop('LongCovid', axis=1)
y = df['LongCovid']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Feature Importance
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)

# Plotting Feature Importance
plt.figure(figsize=(10, 8))
feature_importances.nlargest(20).plot(kind='barh')
plt.title('Top 20 Important Features')
plt.show()

# Display Chi-Squared and Cramér's V results
print("\nChi-Squared p-values:")
for col, p in chi_squared_results.items():
    print(f"{col}: {p}")

print("\nCramér's V results:")
for col, v in cramers_v_results.items():
    print(f"{col}: {v}")
