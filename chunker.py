from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_path = Path('/root/.fastai/data/idwiki')

big_data_energy = [ txt for txt in data_path.rglob('*.txt')]

def chunks(l, n): 
	for i in range(0,len(l), n): 
		yield l[i:i+n]


sentences = [] 
for line in tqdm(big_data_energy): 
	a = open(line).read().splitlines()
	for par in a: 
		if len(par.split())>2: 
			sentences.append(par)


train_path = data_path/'train'
train, y= train_test_split(sentences, test_size=.8, random_state=69)
test,valid = train_test_split(y, test_size=.5, random_state=69)


batches = chunks(train, 5000)
for i, batch in  tqdm(enumerate(batches)): 
	with open(train_path/f'{i}.txt', 'w') as f: 
		for tr in batch: 
			f.writelines(f'{tr}\n')

for i, tr in  tqdm(enumerate(train)): 
	with open(data_path/'test.txt','w') as f: 
		f.writelines(f'{tr}\n')
for i, tr in  tqdm(enumerate(train)): 
	with open(data_path/'valid.txt','w') as f: 
		f.writelines(f'{tr}\n')
