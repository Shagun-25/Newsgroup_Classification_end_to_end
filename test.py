import re
import pandas as pd

df = pd.read_csv(r'E:\Data_Science\Newsgroup_Classification_end_to_end\artifacts\test.csv')
for row in df.itertuples():
    print(repr(row[3]))
    x = row[3]
    break

match = re.findall(r'(Subject:+(.*?)+\n)', row[3])
print(match)