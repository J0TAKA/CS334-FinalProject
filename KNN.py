import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

# Evaluate the model
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
