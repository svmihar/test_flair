from flair.data import Dictionary
from pathlib import Path
import joblib
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
import logging 

logging.basicConfig( format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

Path('../flair_models/backwards/').mkdir(parents=True, exist_ok=True)
# are you training a forward or backward LM?
is_forward_lm = False

# load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')
"""
# get your corpus, process forward and at the character level, then dump to harddisk
"""
# load joblib dump to memory

if Path('../flair_models/backwards/corpus.flair').is_file(): 
	logger.info('corpus found')
	logger.info('now loading the corpus')
	corpus = joblib.load('../flair_models/backwards/corpus.flair')
else: 
	logger.info('making new corpus')
	corpus = TextCorpus('/root/.fastai/data/idwiki/', dictionary, is_forward_lm, character_level=True) 
	logger.info('serializing corpus')
	joblib.dump(corpus, '../flair_models/backwards/corpus.flair')
	logger.info('saving the corpus to ../flair_models')

logger.info('loading corpus done, now creating language model')
# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary, is_forward_lm, hidden_size=2048, nlayers=1) 

if Path('../flair_models/backwards/checkpoint.pt').is_file(): 
	logger.info('checkpoint detected, resuming training')
	trainer = LanguageModelTrainer.load_from_checkpoint('../flair_models/backwards/checkpoint.pt', corpus)
else: 
	# train your language model
	trainer = LanguageModelTrainer(language_model, corpus)

logger.info('we have lift off, good luck ground control')
trainer.train('../flair_models/backwards/', learning_rate=0.1,sequence_length=250, mini_batch_size=150, max_epochs=100, checkpoint=True)
