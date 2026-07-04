import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sales_data.csv")

print("\n===== FIRST 5 RECORDS =====")
print(df.head())

print("\n===== DATASET INFORMATION =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ------------------------
# Correlation Analysis
# ------------------------

numeric_df = df.select_dtypes(include=['number'])

print("\n===== CORRELATION MATRIX =====")
print(numeric_df.corr())

# ------------------------
# Visualization 1
# ------------------------

plt.figure(figsize=(8,5))
sns.barplot(x="Category", y="Sales", data=df)

plt.title("Sales by Category")
plt.show()

# ------------------------
# Visualization 2
# ------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Sales"], bins=5)

plt.title("Sales Distribution")
plt.show()

# ------------------------
# Visualization 3
# ------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(x="Price", y="Sales", data=df)

plt.title("Price vs Sales")
plt.show()

# ------------------------
# Visualization 4
# ------------------------

plt.figure(figsize=(8,5))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully")