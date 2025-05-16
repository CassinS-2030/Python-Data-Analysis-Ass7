# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (using the built-in iris dataset as an example)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Data exploration
print("=== Dataset Overview ===")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\n=== First 5 Rows ===")
print(df.head())
print("\n=== Basic Statistics ===")
print(df.describe())
print("\n=== Species Count ===")
print(df['species'].value_counts())

# Basic analysis
print("\n=== Analysis Results ===")
print(f"Average sepal length: {df['sepal length (cm)'].mean():.2f} cm")
print(f"Maximum petal width: {df['petal width (cm)'].max():.2f} cm")
print("\nAverage measurements by species:")
print(df.groupby('species').mean())

# Visualizations
plt.figure(figsize=(15, 10))

# Histogram of sepal length
plt.subplot(2, 2, 1)
df['sepal length (cm)'].hist(bins=15)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')

# Boxplot of petal width by species
plt.subplot(2, 2, 2)
df.boxplot(column='petal width (cm)', by='species')
plt.title('Petal Width by Species')
plt.suptitle('')  # Remove automatic title
plt.xlabel('Species')
plt.ylabel('Petal Width (cm)')

# Scatter plot of sepal length vs width
plt.subplot(2, 2, 3)
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], 
            c=df['species'].map(colors))
plt.title('Sepal Length vs. Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Bar chart of average petal length by species
plt.subplot(2, 2, 4)
df.groupby('species')['petal length (cm)'].mean().plot(kind='bar')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')

plt.tight_layout()
plt.show()

# Findings and observations
print("\n=== Key Findings ===")
print("1. The dataset contains 150 observations of iris flowers with 4 measurements each.")
print("2. There are equal numbers (50) of each species: setosa, versicolor, and virginica.")
print("3. Setosa flowers tend to be smaller in all measurements compared to the other species.")
print("4. Virginica flowers have the largest petal dimensions on average.")
print("5. There appears to be a positive correlation between sepal length and width.")