import pandas as pd

# Load the dataset
df = pd.read_csv('cow_dataset.csv')

# Extract Lethargy (0/1) - check if 'Lethargy' appears in any symptom column
symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
df['lethargy'] = df[symptom_cols].apply(
    lambda row: 1 if 'Lethargy' in row.values else 0, axis=1
)

# Extract Coughing (0/1) - from the Coughing column
df['coughing'] = df['Coughing'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# Extract Temperature - remove '°C' and convert to float
df['temperature'] = df['Body_Temperature'].str.replace('°C', '', regex=False).astype(float)

# Extract Disease
df['disease'] = df['Disease_Prediction']

# Keep only the required columns
result = df[['lethargy', 'coughing', 'temperature', 'disease']]

# Save to new CSV
result.to_csv('cow_csv_cleaned.csv', index=False)

print(result.head())
print(f"\nTotal rows: {len(result)}")