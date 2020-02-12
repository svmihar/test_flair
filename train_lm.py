from flair.data import Dictionary
import joblib
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
import logging 

logging.basicConfig( format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# are you training a forward or backward LM?
is_forward_lm = True

# load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')

# get your corpus, process forward and at the character level, then dump to harddisk
"""
logger.info('loading the corpus')
corpus = TextCorpus('/root/.fastai/data/idwiki/', dictionary, is_forward_lm, character_level=True) 
logger.info('serializing corpus')
joblib.dump(corpus, 'corpus.flair')
"""

# load joblib dump to memory
logger.info('now loading the corpus')
corpus = joblib.load('corpus.flair')


logger.info('loading corpus done, now creating language model')
# instantiate your language model, set hidden size and number of layers
language_model = LanguageModel(dictionary, is_forward_lm, hidden_size=2048, nlayers=1) 

# train your language model
trainer = LanguageModelTrainer(language_model, corpus)

logger.info('we have lift off, good luck ground control')
trainer.train('../flair_models/', sequence_length=250, mini_batch_size=50, max_epochs=10)
