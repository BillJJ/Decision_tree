import pandas as pd

df = {'a':[1,2,2,3,3,3,3,4,4,4],
      'b':[1,3,7,3,4,5,6,3,2,5],
      'c':[1,8,5,4,6,3,7,2,4,5],
      'd':[2,3,5,2,7,2,3,4,6,4]}

df = pd.DataFrame(df)
y = df.iloc[:,0]
x = df.iloc[:,1:]
s = y.value_counts()
# for a in s.index:
    # print('val:',a,'occurs:',s.loc[a])

preds = pd.Series(index=x.index)
for index, row in df.iterrows():
      preds[index] = row['a']

print(preds)