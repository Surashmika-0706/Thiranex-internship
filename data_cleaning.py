import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data_sales.csv")

print("Original Dataset")
print(df)

# -----------------------------
# Handle Missing Values
# -----------------------------

df["Age"].fillna(df["Age"].mean(), inplace=True)

# -----------------------------
# Remove Duplicates
# -----------------------------

df.drop_duplicates(inplace=True)

# -----------------------------
# Detect Outliers
# -----------------------------

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Salary"] >= lower) &
        (df["Salary"] <= upper)]

print("\nCleaned Dataset")
print(df)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

# -----------------------------
# Visualization 1
# -----------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Salary"], bins=5)
plt.title("Salary Distribution")
plt.savefig("salary_distribution.png")
plt.show()

# -----------------------------
# Visualization 2
# -----------------------------

plt.figure(figsize=(8,5))
sns.boxplot(y=df["Salary"])
plt.title("Salary Outliers")
plt.savefig("salary_boxplot.png")
plt.show()

# -----------------------------
# Visualization 3
# -----------------------------

plt.figure(figsize=(8,5))
sns.countplot(x="Department", data=df)
plt.title("Employees by Department")
plt.savefig("department_count.png")
plt.show()