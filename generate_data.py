import pandas as pd
import numpy as np
import random

# --- Configuration ---
NUM_ROWS = 50000

# --- Column Definitions & Data Generation ---
# Based on the provided screenshot, we'll generate data for each column.

# Demographics
visit_type_categories = ['Outpatient', 'Inpatient', 'Emergency', 'Telehealth']
sexes = ['Female', 'Male']
ethnicities = ['Not Hispanic or Latino', 'Hispanic or Latino']
races = ['White', 'Black or African American', 'Asian', 'American Indian or Alaska Native', 'Other']

# Clinical Conditions (Binary Flags)
# We'll use these as a base to generate other related data.
# Higher probability for more common conditions.
conditions = {
    'HCV': np.random.choice([0, 1], size=NUM_ROWS, p=[0.9, 0.1]),
    'PTSD': np.random.choice([0, 1], size=NUM_ROWS, p=[0.85, 0.15]),
    'Tobacousedisorder': np.random.choice([0, 1], size=NUM_ROWS, p=[0.7, 0.3]),
    'Cannabisusedisorder': np.random.choice([0, 1], size=NUM_ROWS, p=[0.8, 0.2]),
    'Cocaineusedisorder': np.random.choice([0, 1], size=NUM_ROWS, p=[0.95, 0.05]),
    'Alcoholusedisorder': np.random.choice([0, 1], size=NUM_ROWS, p=[0.75, 0.25]),
    'Depression': np.random.choice([0, 1], size=NUM_ROWS, p=[0.65, 0.35]),
    'anxietydisorder': np.random.choice([0, 1], size=NUM_ROWS, p=[0.6, 0.4]),
    'ChronicPain': np.random.choice([0, 1], size=NUM_ROWS, p=[0.55, 0.45]),
}

# Overdoses (Binary Flags)
overdoses = {
    'PriorYearOpioidOverdose': np.random.choice([0, 1], size=NUM_ROWS, p=[0.98, 0.02]),
    'PriorYearNonOpioidOverdose': np.random.choice([0, 1], size=NUM_ROWS, p=[0.97, 0.03]),
}

# Visit & Admission Counts
# Using numpy's random integers for counts
visits = {
    'NumberOfEdVisits': np.random.randint(0, 15, size=NUM_ROWS),
    'NoOfPsychiatryOrPsychologyAdmission': np.random.randint(0, 5, size=NUM_ROWS),
    'TotalHospitalAdmissionsIn180days': np.random.randint(0, 10, size=NUM_ROWS),
    'TotalOutPatientVisitsIn180Days': np.random.randint(0, 30, size=NUM_ROWS),
    'NumberOfEdvisitUniqueEncounter': lambda df: df['NumberOfEdVisits'] - np.random.randint(0, 3, size=NUM_ROWS),
}

# Scores and Continuous Variables
scores = {
    'AgeOnIndexDate': np.random.randint(18, 85, size=NUM_ROWS),
    'PrimaryRUCA_X_x': np.round(np.random.uniform(1.0, 10.0, size=NUM_ROWS), 2),
    'OverallSVI': np.round(np.random.rand(NUM_ROWS), 4),
    'ElixhauserScore': np.random.randint(0, 20, size=NUM_ROWS),
    'NumberOfPrescriptionOpioidInPrior365Days': np.random.randint(0, 50, size=NUM_ROWS),
    'totalMorphineDose': lambda df: df['NumberOfPrescriptionOpioidInPrior365Days'] * np.random.uniform(5, 30),
    'DaysAvailableforStudyAfterIndexDate': np.random.randint(180, 730, size=NUM_ROWS),
}

# Target Variables (Binary)
# These are often the outcomes we want to predict.
# Let's make 'ReceivedMOUD' dependent on a few other factors to create a more realistic relationship.
def generate_received_moud(df):
    # Probability increases with opioid-related history and certain disorders
    base_prob = 0.1
    prob = base_prob + \
           (df['PriorYearOpioidOverdose'] * 0.3) + \
           (df['Alcoholusedisorder'] * 0.1) + \
           (df['ChronicPain'] * 0.05) + \
           (df['totalMorphineDose'] / df['totalMorphineDose'].max() * 0.2)
    
    # Ensure probability is between 0 and 1
    prob = np.clip(prob, 0.01, 0.99)
    
    return (np.random.rand(len(df)) < prob).astype(int)


# --- Assemble the DataFrame ---
print("Generating initial data...")
data = {
    'VisitTypeCategory': [random.choice(visit_type_categories) for _ in range(NUM_ROWS)],
    'Sex': [random.choice(sexes) for _ in range(NUM_ROWS)],
    'Ethnicity': [random.choice(ethnicities) for _ in range(NUM_ROWS)],
    'FirstRace': [random.choice(races) for _ in range(NUM_ROWS)],
}

# Add all generated dictionaries to the main data dictionary
data.update(conditions)
data.update(overdoses)

# Create DataFrame
df = pd.DataFrame(data)

# Add columns that are generated from other columns (using lambda functions)
print("Generating dependent columns...")
for col, func in {**visits, **scores}.items():
    if callable(func):
        df[col] = func(df)
    else:
        df[col] = func

# Clean up any negative values that might have been generated
df['NumberOfEdvisitUniqueEncounter'] = df['NumberOfEdvisitUniqueEncounter'].clip(lower=0)
df['totalMorphineDose'] = df['totalMorphineDose'].clip(lower=0)

# Generate the target variables
print("Generating target variables...")
df['ReceivedMOUD'] = generate_received_moud(df)
df['ReceivedMOUDIn180DaysofOUDDx'] = ((df['ReceivedMOUD'] == 1) & (df['DaysAvailableforStudyAfterIndexDate'] >= 180)).astype(int)


# --- Final Touches ---
# Ensure the column order matches the screenshot
final_column_order = [
    'VisitTypeCategory', 'Sex', 'AgeOnIndexDate', 'Ethnicity', 'FirstRace',
    'PrimaryRUCA_X_x', 'OverallSVI', 'NumberOfEdVisits',
    'NoOfPsychiatryOrPsychologyAdmission', 'PriorYearOpioidOverdose',
    'PriorYearNonOpioidOverdose', 'HCV', 'PTSD', 'Tobacousedisorder',
    'Cannabisusedisorder', 'Cocaineusedisorder', 'Alcoholusedisorder',
    'Depression', 'anxietydisorder', 'ChronicPain', 'ElixhauserScore',
    'NumberOfPrescriptionOpioidInPrior365Days', 'totalMorphineDose',
    'TotalHospitalAdmissionsIn180days', 'TotalOutPatientVisitsIn180Days',
    'DaysAvailableforStudyAfterIndexDate', 'ReceivedMOUDIn180DaysofOUDDx',
    'NumberOfEdvisitUniqueEncounter', 'ReceivedMOUD'
]
df = df[final_column_order]


print(f"\nSuccessfully generated {len(df)} rows of patient data.")

# The final step is to save this data to a file, like a CSV.

df.to_csv('synthetic_patient_data.csv', index=False)
print("Data saved to 'synthetic_patient_data.csv'.")
print("Dataset creation complete.")