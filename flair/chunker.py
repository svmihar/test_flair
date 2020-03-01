from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_path = Path('/root/.fastai/data/idwiki')
def chunks(l, n): 
	for i in range(0,len(l), n): 
		yield l[i:i+n]
sentences = [] 
"""
big_data_energy = [ txt for txt in data_path.rglob('*.txt')]



for line in tqdm(big_data_energy): 
	a = open(line).read().splitlines()
	for par in a: 
		if len(par.split())>2: 
			sentences.append(par)
"""
sentences = open(data_path/'guede.txt').read().splitlines()

train_path = data_path/'train'
Path(train_path).mkdir(exist_ok=True, parents=True)
train, y= train_test_split(sentences, train_size=.8, random_state=69)
test,valid = train_test_split(y, test_size=.5, random_state=69)

print(f"""
train: {len(train)}
test: {len(test)}
valid: {len(valid)}
""")


batches = chunks(train, 7500)
for i, batch in  tqdm(enumerate(batches)): 
	with open(train_path/f'{i}.txt', 'w') as f: 
		for tr in batch: 
			f.writelines(f'{tr}\n')

for i, tr in  tqdm(enumerate(test)): 
	with open(data_path/'test.txt','w') as f: 
		f.writelines(f'{tr}\n')
for i, tr in  tqdm(enumerate(valid)): 
	with open(data_path/'valid.txt','w') as f: 
		f.writelines(f'{tr}\n')
