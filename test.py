import numpy as np
import pandas as pd

data = np.zeros((2,3),float)

print(data[1,2])
df = pd.read_csv('./data/train.csv')
# df = df['微博图片'].fillna('无',inplace=True)
# df = df['微博视频'].fillna('无',inplace=True)
for i in range(1,10):
    if df.iloc[i,4] !='[]':
        print(df.iloc[i,4])
        data[0,df.iloc[i,6]+1] = str(int(data[0,df.iloc[i,6]+1]) + 1)
        print(data[0,df.iloc[i,6]+1])
    if df.iloc[i, 5] != ' ':
        data[1, df.iloc[i, 6] + 1] = str(int(data[1, df.iloc[i, 6] + 1]) + 1)

print(data)