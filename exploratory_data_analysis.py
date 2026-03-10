# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from ydata_profiling import ProfileReport

# Reading the data file
file_path = pathlib.Path(r"C:\Users\rathi\Downloads\S01 - Machine-Learning-A-Z-Codes-Datasets\Machine-Learning-A-Z-Codes-Datasets\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv")
df = pd.read_csv(file_path)

# Displaying whole dataset
print(df, end='\n \n')

# Inspecting data structure
print(f"Shape:", df.shape, end='\n \n')
print(df.head(), end='\n \n')
print(df.tail(), end='\n \n')

# Inspecting data types
print(df.info(), end='\n \n')

# Summary stats
print(df.describe(), end='\n \n')
print(df.describe(include='object'), end='\n \n')

# Missing and duplicated values 
print(df.isna(), end='\n \n') # isna and isnull function do the same work
print(df.isnull(), end='\n \n')
print(df.isnull().sum(), "\n", df.isna().sum(), end='\n \n')
print(df.duplicated(), "\nDuplicated rows count:", df.duplicated().sum(), end='\n \n')

# Unique values
print(df.value_counts())
print(df['Age'].value_counts())


# ---------------------------------------------Visualization-----------------------------------------------------
df['Age'].hist()
plt.show()
df['Salary'].plot(kind='bar')
plt.show()


# Creating an EDA report
profile = ProfileReport(df)
profile.to_file("eda_report.html")