#PackageInstallation
pip install pandas numpy matplotlib seaborn scikit-learn

#importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#LoadingDS
df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())

#ExploreDS
print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

#HandlingMissingValues
#FillAgeByMedian
df['Age'].fillna(df['Age'].median(), inplace=True)
#FillEmbarkedByMode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
#FillCabinByUnknown
df['Cabin'].fillna("Unknown", inplace=True)
print(df.isnull().sum())

#ConvertingCategoricalDateToNumerical
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
print(df.head())

#RemovingUnnecessaryColumns
df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

#DetectOutliers
plt.figure(figsize=(10,5))

sns.boxplot(data=df[['Age','Fare']])

plt.title("Boxplot for Outlier Detection")
plt.savefig("boxplot_outliers.png", dpi=300, bbox_inches='tight')
plt.show()

#RemoveOutliersUsingIQRMethod
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

print("Shape after removing outliers:")
print(df.shape)

#Standardization
scaler = StandardScaler()

df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])
print(df.head())

#SaveCleanedData
df.to_csv("cleaned_titanic.csv", index=False)

print("Cleaned dataset saved successfully!")

#Visualization
#SurvivalCount
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png", dpi=300, bbox_inches='tight')
plt.show()
#correlationHeatmap
plt.figure(figsize=(8,6))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
plt.show()