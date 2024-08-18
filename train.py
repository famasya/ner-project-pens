from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from datetime import datetime

# define columns
columns = {0: "text", 1: "ner"}

# this is the folder in which train, test and dev files reside
data_folder = "splits/2024-06-12T07:48:09.778Z"

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train.txt",
    test_file="test.txt",
    dev_file="valid.txt",
)

label_dict = corpus.make_label_dictionary(label_type="ner", add_unk=False)
# model = "distilbert-base-multilingual-cased"
# model = "xlm-roberta-base"
model = "Unbabel/xlm-roberta-comet-small"
embeddings = TransformerWordEmbeddings(
    model=model,
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
    allow_long_sentences=True,
)

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type="ner",
    use_crf=False,
    use_rnn=False,
)

trainer = ModelTrainer(tagger, corpus)

# start the training
date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
trainer.fine_tune(
    "models/{}-{}".format(model.replace("/", "-"), date),
    max_epochs=10,
    mini_batch_size=8,
    mini_batch_chunk_size=1,  # remove this parameter to speed up computation if you have a big GPU
)
