import pandas as pd

data = [[1, 2], [3, 4]]
df = pd.DataFrame(data, columns=range(0,len(data)), index=range(0, len(data)))
print(df)