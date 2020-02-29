from pathlib import Path 
from tokenizers import ByteLevelBPETokenizer

data = Path('/root/.train_lm/')
paths = [str(x) for x in data.glob("**/*.txt")]
tokenizer = ByteLevelBPETokenizer()


if __name__ == "__main__": 
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        '<s>', 
        '<pad>', 
        '</s>', 
        '<unk>', 
        '<mask>'])
    tokenizer.save('berto')
