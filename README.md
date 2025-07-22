# RTEC Training: Python for Data Analysis

> **A comprehensive training on Python for Data Analysis - From Fundamentals to Machine Learning**

Prepared by: **Nahiyan Bin Noor, MS**  
Data Analyst - Intermediate  
Institute for Digital Health & Innovation  
University of Arkansas for Medical Sciences

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Training Materials](#training-materials)
- [Topics Covered](#topics-covered)
- [Sample Code](#sample-code)
- [Resources](#resources)
- [Contact](#contact)

##  Overview

This repository contains comprehensive training materials for learning Python for data analysis, specifically designed for healthcare data professionals. The training covers everything from Python basics to advanced machine learning and statistical modeling techniques.

**Key Learning Outcomes:**
- Master Python fundamentals for data analysis
- Learn to manipulate and visualize healthcare data
- Understand predictive and inferential modeling
- Apply machine learning techniques to real-world scenarios

##  Prerequisites

- Basic understanding of statistics
- Familiarity with healthcare data concepts
- No prior programming experience required

##  Setup Instructions

Follow these steps to set up your Python environment for the training:

### Step 1: Install Python
1. Go to [python.org/downloads](https://www.python.org/downloads)
2. Click 'Download Python 3.x.x' (latest version)
3. **Important:** Check 'Add Python to PATH' before installing

### Step 2: Install Anaconda
1. Go to [anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
2. Download installer for your operating system
3. Install with default settings
4. This provides Python + conda package manager

### Step 3: Install Visual Studio Code
1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Run installer with default options
3. Check 'Add to PATH' and 'Register Code as editor'

### Step 4: Create & Activate Conda Environment
Open Anaconda Prompt or terminal:
```bash
# Create environment
conda create -n myenv python=3.11

# Activate environment
conda activate myenv
```

### Step 5: Install Required Packages
```bash
# Core data science packages
conda install pandas numpy matplotlib seaborn statsmodels scikit-learn

# Alternative using pip
pip install pandas numpy matplotlib seaborn

# Install Jupyter for interactive notebooks
conda install jupyter
```

### Step 6: Setup VS Code for Python
1. Open VS Code
2. Press `Ctrl+Shift+P`
3. Type 'Python: Select Interpreter'
4. Choose the interpreter with `(myenv)`

##  Training Materials

This repository includes:

- **Setup Guide** (`Setup Guide Python Conda VSCode.docx`) - Step-by-step installation instructions
- **Presentation Slides** (`RTEC_Data_Analyst_Training_Slide_By_Nahiyan.pdf`)
- **Create Synthetic Dataset**(`generate_data.py`) - Generated healthcare data for practice
- **Synthetic Dataset**(`synthetic_patient_data.csv`) - Generated healthcare data for practice
- **Sample Code**(`analysis.ipynb`) - Python scripts demonstrating key concepts
- 
##  Topics Covered

### 1. Introduction to Python
- What is Python and why use it for data analysis?
- Python ecosystem for data science
- Core libraries: Pandas, Matplotlib, Seaborn, Scikit-learn

### 2. Data Generation & Manipulation
- Creating synthetic healthcare datasets
- Data importing and first exploration
- Filtering and selecting data subsets

### 3. Exploratory Data Analysis (EDA)
- Descriptive statistics
- Data grouping and aggregation
- Identifying patterns in healthcare data

### 4. Data Visualization
- Numerical distributions (histograms)
- Categorical distributions (count plots)
- Creating publication-ready plots

### 5. Machine Learning Models
#### Prediction Models:
- **Logistic Regression** - Binary outcome prediction
- **Random Forest** - Advanced ensemble methods

#### Inference Models:
- **Linear Regression** - Understanding continuous outcomes
- **Poisson Regression** - Modeling count data (e.g., ED visits)
- **Negative Binomial Regression** - Handling overdispersed count data
- **Logistic Regression** - Binary outcome inference

### 6. Statistical Modeling Comparison
Understanding when to use:
- OLS vs Poisson vs Negative Binomial regression
- Model interpretation and selection criteria

##  Sample Code

### Data Generation Example
```python
import pandas as pd
import numpy as np
import random

NUM_ROWS = 50000
sexes = ['Female', 'Male']
races = ['White', 'Black or African American', 'Asian', 'Other']

data = {
    'Sex': [random.choice(sexes) for _ in range(NUM_ROWS)],
    'AgeOnIndexDate': np.random.randint(18, 85, size=NUM_ROWS),
    'FirstRace': [random.choice(races) for _ in range(NUM_ROWS)],
    'Depression': np.random.choice([0, 1], size=NUM_ROWS, p=[0.7, 0.3]),
    'ChronicPain': np.random.choice([0, 1], size=NUM_ROWS, p=[0.6, 0.4]),
    'ElixhauserScore': np.random.randint(0, 20, size=NUM_ROWS),
    'NumberOfEdVisits': np.random.randint(0, 15, size=NUM_ROWS)
}

df = pd.DataFrame(data)
df.to_csv('synthetic_patient_data.csv', index=False)
```

### Basic EDA Example
```python
import pandas as pd

# Load data
df = pd.read_csv('synthetic_patient_data.csv')

# Display first 5 rows
print(df.head())

# Get summary statistics
print(df.describe())

# Value counts for categorical variables
print(df['FirstRace'].value_counts())

# Group by and aggregate
avg_age_by_sex = df.groupby('Sex')['AgeOnIndexDate'].mean()
print(avg_age_by_sex)
```

## 📊 Key Libraries Used

- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Basic plotting
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning
- **Statsmodels** - Statistical modeling

## 📖 Additional Resources

- [Python for Data Analysis by Wes McKinney](https://wesmckinney.com/book/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Statsmodels Documentation](https://www.statsmodels.org/)


## 📧 Contact

**Nahiyan Bin Noor, MS**  
Data Analyst - Intermediate  
Institute for Digital Health & Innovation  
University of Arkansas for Medical Sciences

For questions about this training or the materials, please open an issue in this repository.

## 📄 License

This training material is provided for educational purposes. Please cite appropriately if using in academic or professional settings.

---

*Happy coding and analyzing! 🐍*
