import pandas as pd

# Replace 'largefile.csv' with the name of your file
file_name = 'test_scans.csv'
num_parts = 5

# Read the file into a DataFrame
df = pd.read_csv(file_name)

# Determine the number of rows in each part
rows_per_part = len(df) // num_parts

# Split the DataFrame and save each part
for i in range(num_parts):
    start_row = i * rows_per_part
    # Make sure the last part includes any remaining rows
    if i == num_parts - 1:
        end_row = len(df)
    else:
        end_row = (i + 1) * rows_per_part

    part_df = df[start_row:end_row]
    part_df.to_csv(f'part_{i+1+12}.csv', index=False)

print(f"File split into {num_parts} parts successfully.")

