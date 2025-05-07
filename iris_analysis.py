# Task: Data Analysis and Visualization with Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

    # Cleaning: Drop missing values (if any)
    df.dropna(inplace=True)

except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
print("\nBasic statistics:")
print(df.describe())

print("\nMean of numerical columns grouped by species:")
print(df.groupby("species").mean())

# Task 3: Data Visualization

sns.set(style="whitegrid")

# Line Chart: Mean petal length for each species (simulated trend over time for demo)
plt.figure(figsize=(8, 5))
for species in df['species'].unique():
    species_data = df[df['species'] == species].reset_index()
    plt.plot(species_data.index, species_data['petal length (cm)'], label=species)
plt.title("Petal Length Trends per Species")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(7, 5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(7, 5))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(7, 5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# Findings
print("\nObservations:")
print("- Iris-virginica generally has the highest petal length.")
print("- Petal length increases consistently across species from setosa to virginica.")
print("- Sepal width is fairly normally distributed.")
print("- There is a clear positive correlation between sepal length and petal length.")

