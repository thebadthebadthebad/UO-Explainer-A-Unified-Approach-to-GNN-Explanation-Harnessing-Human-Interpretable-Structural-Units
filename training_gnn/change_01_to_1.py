import os
print(f"Current working directory: {os.getcwd()}")

data_name = 'bashapes'

data_path = f'../dataset/{data_name}/'

with open(f'{data_path}feature.txt', 'r') as s, open(f'{data_path}features.txt', 'w') as c:
    for i in s.readlines():
        i = i.replace('0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 ', '1 1 1 1 1 1 1 1 1 1 ')
        c.writelines(i)

