import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Load the dataset
df = pd.read_csv('LongCovidData.csv')

# Preprocess the data (handling missing values, encoding, etc.)
# Dropping the index column
df = df.drop(columns=['Unnamed: 0'])
# Recoding LongCovid to Binary Representation 2 = false 1 = true
df['LongCovid'] = df['LongCovid'].apply(lambda x: 0 if x == 2 else 1)

sns.set()
plt.figure(figsize=(20, 15))
ax = plt.subplot(111)
# Calculate the correlation matrix
corrMat = df.corr(method='pearson')
# Create the heatmap
ax = sns.heatmap(corrMat, annot=True, annot_kws={"fontsize": 8})
# Export the heatmap
outpath = 'Train_heatmap.png'
plt.savefig(outpath, dpi=600)


if __name__ == "__main__":
    main()


