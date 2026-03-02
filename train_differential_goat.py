import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the Goat CSV
print("Loading Goat Data...")
df = pd.read_csv('goat_csv_cleaned.csv')

# 2. Harmonize Names (Standardization for Goats)
# This groups similar strings and uses standard vet abbreviations
mapping = {
    "Caprine Arthritis Encephalitis": "CAE",
    "Caprine Arthritis Encephalitis Virus": "CAE",
    "Caprine Arthritis": "CAE",
    "Caprine Viral Arthritis": "CAE",
    "Caprine Respiratory Disease": "CRD",
    "Caprine Pleuropneumonia": "CCPP",
    "Foot-and-Mouth Disease": "FMD",
    "Gastrointestinal Infection": "GI Infection"
}
df['disease'] = df['disease'].replace(mapping)

# 3. Data Augmentation (Expanding for stability)
print("Augmenting data points with thermal jitter...")
expanded = []
for _ in range(100):
    for _, row in df.iterrows():
        # Adds tiny biological temperature variations so the model doesn't overfit
        jitter = round(np.random.normal(0, 0.05), 2)
        expanded.append([row['lethargy'], row['coughing'], round(row['temperature'] + jitter, 1), row['disease']])

final_df = pd.DataFrame(expanded, columns=['lethargy', 'coughing', 'temperature', 'disease'])

# 4. Training
print("Training the Random Forest Agent...")
X = final_df[['lethargy', 'coughing', 'temperature']]
y = final_df['disease']

model = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=42)
model.fit(X, y)

# 5. Save Model and the Class Names for the App
joblib.dump(model, 'clinical_model_goat.pkl')
joblib.dump(model.classes_, 'disease_classes_goat.pkl') 

print(f"✅ Goat Differential Model Saved!")
print(f"🩺 It is now trained to rank {len(model.classes_)} distinct Caprine conditions:")
print(list(model.classes_))
