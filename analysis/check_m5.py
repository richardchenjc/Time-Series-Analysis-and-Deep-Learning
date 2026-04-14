import pandas as pd

s = pd.read_csv('results/summary_table.csv')
m5 = s[s['dataset'] == 'M5'].copy()

print('M5 summary:')
print(m5[['dataset', 'model', 'mase_mean', 'mase_std', 'mae_mean', 'mae_std']].to_string(index=False))