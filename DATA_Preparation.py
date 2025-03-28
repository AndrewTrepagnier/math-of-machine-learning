import numpy as np
import os,sys
from util_DATA_Prep import *
import pandas as pd

file_ELP = '/Users/andrewtrepagnier/.cursor-tutor/projects/MathMachineLearning/Data_preparation_techniques/dataFiles/data-ECommerce-Labor_Prod.xlsx'
file_TS  = '/Users/andrewtrepagnier/.cursor-tutor/projects/MathMachineLearning/Data_preparation_techniques/dataFiles/data-Total-Sale.xlsx'

#--------------------------------------------------
# Read Excel files
#--------------------------------------------------
# DATA_Ecommerce_labor, header_Ecommerce_labor = pd.read_excel(file_ELP)
# DATA_Total_sale,  header_Total_sale  = pd.read_excel(file_TS)

df_elp = pd.read_excel(file_ELP)
df_ts = pd.read_excel(file_TS)

# Print column names to see what we're working with
print("TS columns:", df_ts.columns)

# Sort df_elp by NAICS code first, then by year in ascending order
df_elp = df_elp.sort_values(by=['NAICS', 'year'], ascending=[True, True])

# First, insert empty 'Total Sale' column
df_elp.insert(2, 'Total Sale', np.nan)

# Create a list to store all values in sequence
big_vector = []

# Loop through rows 0 to 20
for i in range(0, 21):
    current_row = df_ts.iloc[i, 1:].values
    big_vector.extend(current_row)

# Convert to numpy array
big_vector = np.array(big_vector)
repeated_values = np.tile(big_vector, len(df_elp) // len(big_vector) + 1)[:len(df_elp)]
df_elp['Total Sale'] = repeated_values

# Save to Excel file
df_elp.to_excel('processed_data.xlsx', index=False)

print("File saved as 'processed_data.xlsx'")

# Print first and last year for each NAICS code
for naics in df_elp['NAICS'].unique():
    naics_data = df_elp[df_elp['NAICS'] == naics]
    first_year = naics_data.iloc[0]
    last_year = naics_data.iloc[-1]
    print(f"\nNAICS {naics}:")
    print(f"1999: {first_year['Total Sale']}")
    print(f"2021: {last_year['Total Sale']}")

#--------------------------------------------------



# Combine and Sort
#  Combine the above for <DATA> and <header>
#  in the order ['NAICS', 'year', 'Total', 'E-commerce', 'Labor-Prod']
#  Sort: First, with 'NAICS code' and then with 'year'
#--------------------------------------------------

# Implement a function or two into "util_DATA_Prep.py" to complete

#--------------------------------------------------
# You can save the trimmed "DATA" to an Excel file:
# First, you should get combined <DATA> and <header>
#--------------------------------------------------

