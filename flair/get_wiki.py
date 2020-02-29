from fastai import *
from fastai.text import *
from nlputils import split_wiki,get_wiki
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


bs= 128
data_path = Config.data_path()
lang = 'id'
name = f'{lang}wiki'
path = data_path/name
path.mkdir(exist_ok=True, parents=True)
lm_fns = [f'{lang}_wt', f'{lang}_wt_vocab']
logger.info('now downloading wiki')
get_wiki(path,lang)
logger.info(f'wiki downladed\n{path.ls()}')


logger.info('now creating data_lm')
dest = split_wiki(path,lang)
"""
data = (TextList.from_folder(dest)
            .split_by_rand_pct(0.1, seed=42)
            .label_for_lm()           
            .databunch(bs=bs, num_workers=8))

data.save(f'{lang}_tmp')
logger.info(f'{len(data.vocab.itos)},{len(data.train_ds)}')

data = load_data(path, f'docs/{lang}_tmp', bs=bs)

"""
