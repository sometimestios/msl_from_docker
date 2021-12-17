import pandas as pd

df=pd.DataFrame([0,1,2,3,4,5])
r=df.ewm(alpha=0.9).mean()
print(r)