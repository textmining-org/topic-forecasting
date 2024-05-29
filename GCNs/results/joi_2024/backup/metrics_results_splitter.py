import pandas as pd

df = pd.read_csv('metrics_trng.csv')

df_agcrn = df[df['model'] == 'agcrn']
df_agcrn['K'] = 2
df_agcrn['embedd_dim'] = 4
df_agcrn['out_channels'] = 32
df_agcrn.to_csv('metrics_trng_agcrn.csv', index=False)

df_a3tgcn2 = df[df['model'] == 'a3tgcn2']
df_a3tgcn2['out_channels'] = 32
df_a3tgcn2.to_csv('metrics_trng_a3tgcn2.csv', index=False)
