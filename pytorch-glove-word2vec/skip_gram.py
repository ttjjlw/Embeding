import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm

from tools import CorpusPreprocess, VectorEvaluation

# params
class SkipGram(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size, neg_model=True):
        super(SkipGram, self).__init__()
        self.neg_model = neg_model
        if not self.neg_model:
            self.embedding_matrix = torch.nn.Embedding(vocab_size*2-1, embedding_dim)
            torch.nn.init.xavier_uniform_(self.embedding_matrix.weight.data)
        else:
            self.v_embedding_matrix = torch.nn.Embedding(vocab_size,
                                                         embedding_dim)

            torch.nn.init.xavier_uniform_(self.v_embedding_matrix.weight.data)
            
            
            self.u_embedding_matrix = torch.nn.Embedding(vocab_size,
                                                         embedding_dim)
            self.u_embeddings.weight.data.uniform_(-0, 0)
    def forward(self, pos_v, pos_u, neg_v, neg_u):

        if not self.neg_model:
            pos_v = self.embedding_matrix(pos_v)
            pos_u = self.embedding_matrix(pos_u)
            neg_v = self.embedding_matrix(neg_v)
            neg_u = self.embedding_matrix(neg_u)
        else:
            pos_v = self.v_embedding_matrix(pos_v)
            pos_u = self.u_embedding_matrix(pos_u)
            neg_v = self.v_embedding_matrix(neg_v)
            neg_u = self.u_embedding_matrix(neg_u)

        pos_z = torch.sum(pos_v.mul(pos_u), dim = 1,keepdim=True)
        neg_z = torch.sum(neg_v.mul(neg_u), dim = 1,keepdim=True)

        pos_a = torch.nn.functional.logsigmoid(pos_z)
        neg_a = torch.nn.functional.logsigmoid(-1 * neg_z)

        pos_loss = torch.sum(pos_a)
        neg_loss = torch.sum(neg_a)

        loss = -1 * (pos_loss + neg_loss)
        return loss
