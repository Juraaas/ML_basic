import pandas as pd

input_file = 'australian.tab'
output_file = 'australian.csv'

data_lines = []

with open(input_file, 'r') as file:
    lines = file.readlines()
    start_data = False

    for line in lines:
        if start_data:
            fields = line.strip().split()
            data_lines.append(fields)
        elif line.strip() == "OBJECTS 690":
            start_data = True

header = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11",
    "A12", "A13", "A14", "CLASS"
]

df = pd.DataFrame(data_lines, columns=header)

df["CLASS"] = df["CLASS"].replace({'+': 1, '-': 0})

df.to_csv(output_file, index=False)

data = pd.read_csv(output_file)

# Find Available Decision Classes
decision_classes = data['CLASS'].unique()
print("Available Decision Classes:", decision_classes)
print("Number of Decision Classes:", len(decision_classes))

# Size of Decision Classes
class_sizes = data['CLASS'].value_counts()
print("Size of Decision Classes:")
print(class_sizes)

# Minimal and Maximal Values for Each Numerical Attribute
numerical_attributes = data.select_dtypes(include='number')
min_values = numerical_attributes.min()
max_values = numerical_attributes.max()
print("Minimal Values for Each Numerical Attribute:")
print(min_values)
print("\nMaximal Values for Each Numerical Attribute:")
print(max_values)

# Custom Standard Deviation Calculation
def custom_std_deviation(column_values):
    mean = sum(column_values) / len(column_values)
    squared_diff = sum((x - mean) ** 2 for x in column_values)
    std_dev = (squared_diff / len(column_values)) ** 0.5
    return std_dev

for column in numerical_attributes.columns:
    std_dev = custom_std_deviation(data[column])
    print(f"Standard Deviation for {column}: {std_dev}")

# Handling Missing Values with Custom Methods
columns_with_missing = ['A1', 'A2', 'A4', 'A5', 'A6', 'A7', 'A14']

def fill_missing_with_custom_method(column):
    mean = sum(column) / len(column)
    return [mean if pd.isnull(x) else x for x in column]

for column in columns_with_missing:
    data[column] = fill_missing_with_custom_method(data[column])

# Normalization into Intervals
# Assuming 'numerical_attributes' contains the numerical columns for normalization
# Custom normalization method
def custom_normalize(column, a, b):
    min_val = min(column)
    max_val = max(column)
    return [((x - min_val) * (b - a) / (max_val - min_val) + a) for x in column]

for column in numerical_attributes.columns:
    data[column] = custom_normalize(data[column], -1, 1)  # Adjust interval as needed

# Standardization
# Custom method for standardization
def custom_standardize(column):
    mean = sum(column) / len(column)
    variance = sum((x - mean) ** 2 for x in column) / len(column)
    return [(x - mean) / variance ** 0.5 for x in column]

for column in numerical_attributes.columns:
    data[column] = custom_standardize(data[column])


dummy_vars = pd.get_dummies(data['A8'], prefix='A8')
data = pd.concat([data, dummy_vars], axis=1)
data.drop('A8', axis=1, inplace=True)  # Dropping the original column

print(data['A1'])
print(data['A8_-1.0475039081979491'])

