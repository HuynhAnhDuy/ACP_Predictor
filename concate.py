import pandas as pd

# Load the first CSV normally
df1 = pd.read_csv('ACP_onehot.csv')

# Load the second CSV and drop its first column (by index)
df2 = pd.read_csv('ACP_esm.csv').iloc[:, 1:]

# Concatenate them along the rows (or use axis=1 for side-by-side)
result = pd.concat([df1, df2], axis=1)  # axis=0 means stacking vertically

# Save or display
result.to_csv('ACP_approved_new_onehot_esm.csv', index=False)
print(result)