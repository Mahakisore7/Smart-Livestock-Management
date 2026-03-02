import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1. Load your CSV
df = pd.read_csv('cow_csv_cleaned.csv')

# 2. Harmonize Names (Standardization)
mapping = {
    "Bovine Respiratory Disease": "BRD", "Bovine Respiratory Disease Complex": "BRD",
    "Bovine Viral Diarrhea": "BVD", "Bovine Mastitis": "Mastitis",
    "Bovine Tuberculosis": "Tuberculosis", "Bovine Johne's Disease": "Johne's Disease",
    "Bovine Respiratory Syncytial Virus": "RSV", "Respiratory Syncytial Virus": "RSV",
    "Bovine Pneumonia": "Pneumonia", "Bovine Influenza": "Influenza", "Bovine Parainfluenza": "RSV"
}
df['disease'] = df['disease'].replace(mapping)

# 3. Data Augmentation (Expanding for stability)
expanded = []
for _ in range(100):
    for _, row in df.iterrows():
        jitter = round(np.random.normal(0, 0.05), 2)
        expanded.append([row['lethargy'], row['coughing'], round(row['temperature'] + jitter, 1), row['disease']])

final_df = pd.DataFrame(expanded, columns=['lethargy', 'coughing', 'temperature', 'disease'])

# 4. Training
X = final_df[['lethargy', 'coughing', 'temperature']]
y = final_df['disease']

model = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=42)
model.fit(X, y)

# 5. Save Model and the Class Names (The list of diseases)
joblib.dump(model, 'clinical_model.pkl')
joblib.dump(model.classes_, 'disease_classes.pkl') # We need this to label the probabilities!

print(f"✅ Differential Model Saved. It can now rank {len(model.classes_)} diseases.")
