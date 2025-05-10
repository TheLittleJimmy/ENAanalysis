import pandas as pd

# Read the ENA data CSV file with UTF-8 encoding
df = pd.read_csv('ena_data.csv', encoding='utf-8')

# Extract all codes from the 'code' column
all_codes = []
for codes in df['code']:
    if isinstance(codes, str):  # Check if the value is a string
        code_values = [code.strip() for code in codes.split(',')]
        all_codes.extend(code_values)

# Create a list of unique codes
code_list = sorted(list(set(all_codes)))
print(f"Unique codes found: {code_list}")

# Add zero columns for each unique code
for code in code_list:
    df[code] = 0

# Set values to 1 where the code appears in the 'code' column
for index, row in df.iterrows():
    if isinstance(row['code'], str):
        row_codes = [code.strip() for code in row['code'].split(',')]
        for code in row_codes:
            if code in code_list:
                df.at[index, code] = 1

# Display the modified DataFrame
print("\nDataFrame with updated columns (showing first 5 rows):")
print(df.head())

# Save the modified DataFrame with UTF-8-BOM encoding for Excel compatibility
df.to_csv('enadata_processed.csv', index=False, encoding='utf-8-sig')
print("\nProcessed data saved to 'ena_data_processed.csv'") 