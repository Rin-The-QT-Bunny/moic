import torch
import torch.nn as nn

from aluneth.rinlearn.nlp.utils import *

class VectorConstructor(nn.Module):
    def __init__(self,word_dim,corpus):
        super().__init__()
        # construct vector embeddings and other utils for given corpus
        self.corpus = corpus
        self.key_words = []
        self.token_to_id = build_vocab(corpus)
        self.id_to_token = reverse_diction(self.token_to_id)
        self.word_vectors = nn.Embedding(9999,word_dim)

    def forward(self,sentence):
        code = encode(tokenize(sentence),self.token_to_id)
        word_vectors = self.word_vectors(torch.tensor(code))
        return word_vectors
    
    def get_word_vectors(self,sentence):
        return sentence