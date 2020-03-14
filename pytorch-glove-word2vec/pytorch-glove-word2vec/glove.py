import torch, pickle, os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm

from tools import CorpusPreprocess, VectorEvaluation

# params
x_max = 100
alpha = 0.75
epoches = 1
min_count = 0
batch_size = 512
windows_size = 5
vector_size = 300
learning_rate = 0.001
glove_path = 'glmodel/'
if not os.path.exists(glove_path): os.makedirs(glove_path)
model_name = 'glove_{}.pkl'.format(str(vector_size))
glove_model_file = os.path.join(glove_path, model_name)
# print('模型保存路径：',glove_model_file)
# get gpu
use_gpu = torch.cuda.is_available()


# calculation weight
def fw(X_c_s):
    return (X_c_s / x_max) ** alpha if X_c_s < x_max else 1


class Glove(nn.Module):
    def __init__(self, vocab_size, vector_size):
        super(Glove, self).__init__()
        # center words weight and biase
        self.c_weight = nn.Embedding(len(vocab_size), vector_size,
                                     _weight=torch.randn(len(vocab_size),
                                                         vector_size,
                                                         dtype=torch.float,
                                                         requires_grad=True) / 100)

        self.c_biase = nn.Embedding(len(vocab_size), 1, _weight=torch.randn(len(vocab_size),
                                                                            1, dtype=torch.float,
                                                                            requires_grad=True) / 100)

        # surround words weight and biase
        self.s_weight = nn.Embedding(len(vocab_size), vector_size,
                                     _weight=torch.randn(len(vocab_size),
                                                         vector_size, dtype=torch.float,
                                                         requires_grad=True) / 100)

        self.s_biase = nn.Embedding(len(vocab_size), 1,
                                    _weight=torch.randn(len(vocab_size),
                                                        1, dtype=torch.float,
                                                        requires_grad=True) / 100)

    def forward(self, c, s):
        c_w = self.c_weight(c)
        c_b = self.c_biase(c)
        s_w = self.s_weight(s)
        s_b = self.s_biase(s)
        return torch.sum(c_w.mul(s_w), 1, keepdim=True) + c_b + s_b


# read data
class TrainData(Dataset):
    def __init__(self, coo_matrix):
        self.coo_matrix = [((i, j), coo_matrix.data[i][pos]) for i, row in enumerate(coo_matrix.rows) for pos, j in
                           enumerate(row)]

    def __len__(self):
        return len(self.coo_matrix)

    def __getitem__(self, idex):
        sample_data = self.coo_matrix[idex]
        sample = {"c": sample_data[0][0],
                  "s": sample_data[0][1],
                  "X_c_s": sample_data[1],
                  "W_c_s": fw(sample_data[1])}
        return sample


def loss_func(X_c_s_hat, X_c_s, W_c_s):
    X_c_s = X_c_s.view(-1, 1)
    W_c_s = X_c_s.view(-1, 1)
    loss = torch.sum(W_c_s.mul((X_c_s_hat - torch.log(X_c_s)) ** 2))
    return loss


# save vector
def save_word_vector(file_name, corpus_preprocessor, glove):
    with open(file_name, "w", encoding="utf-8") as f:
        if use_gpu:
            c_vector = glove.c_weight.weight.data.cpu().numpy()
            s_vector = glove.s_weight.weight.data.cpu().numpy()
            vector = c_vector + s_vector
        else:
            c_vector = glove.c_weight.weight.data.numpy()
            s_vector = glove.s_weight.weight.data.numpy()
            vector = c_vector + s_vector
        # try:
        #     with open('output/vector.pkl', 'wb') as p:
        #         pickle.dump(vector, p)
        #     print('vector的shape', vector.shape)
        # except:
        #     print('打印vector的shape有误')
        for i in tqdm(range(len(vector))):
            word = corpus_preprocessor.idex2word[i]
            s_vec = vector[i]
            s_vec = [str(s) for s in s_vec.tolist()]
            write_line = word + " " + " ".join(s_vec) + "\n"
            f.write(write_line)
        print("Glove vector save complete!")


def train_model(epoches, corpus_file_name):
    corpus_preprocessor = CorpusPreprocess(corpus_file_name, min_count)
    coo_matrix = corpus_preprocessor.get_cooccurrence_matrix(windows_size)
    vocab = corpus_preprocessor.get_vocab()
    glove = Glove(vocab, vector_size)

    print(glove)
    if os.path.isfile(glove_model_file):
        glove.load_state_dict(torch.load(glove_model_file))
        print('载入模型{}'.format(glove_model_file))
    if use_gpu:
        glove.cuda()
    optimizer = torch.optim.Adam(glove.parameters(), lr=learning_rate)

    train_data = TrainData(coo_matrix)
    data_loader = DataLoader(train_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)

    steps = 0
    for epoch in range(epoches):
        print(f"currently epoch is {epoch + 1}, all epoch is {epoches}")
        avg_epoch_loss = 0
        for i, batch_data in enumerate(data_loader):
            c = batch_data['c']
            s = batch_data['s']
            X_c_s = batch_data['X_c_s']
            W_c_s = batch_data["W_c_s"]

            if use_gpu:
                c = c.cuda()
                s = s.cuda()
                X_c_s = X_c_s.cuda()
                W_c_s = W_c_s.cuda()

            W_c_s_hat = glove(c, s)
            loss = loss_func(W_c_s_hat, X_c_s, W_c_s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_epoch_loss += loss / len(train_data)
            if steps % 1000 == 0:
                print(f"Steps {steps}, loss is {loss.item()}")
            steps += 1
        print(f"Epoches {epoch + 1}, complete!, avg loss {avg_epoch_loss}.\n")
    save_word_vector(save_vector_file_name, corpus_preprocessor, glove)
    torch.save(glove.state_dict(), glove_model_file)

def get_dict_and_embedding(model_dir,embed_dir,vocabulary_dir,embed_dim):
    word2id={'<pad>':0}
    index=1
    embeddings=[]
    embeddings.append([0]*embed_dim)
    with open (model_dir,'r', encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            if len(line.strip().split())<3:
                continue
            word=line.strip().split()[0]
            data=line.strip().split()[1:]
            word2id[word]=index
            index+=1
            try:
                assert len(data)==embed_dim
            except:
                print(len(data),embed_dim)
            embeddings.append(data)
    embeddings=np.array(embeddings)
    print('glove embeddings的shape为：({}*{})=={}'.format(index,embed_dim,embeddings.shape))
    with open(embed_dir,'wb') as p:
        pickle.dump(embeddings,p)
    with open(vocabulary_dir,'wb') as p1:
        pickle.dump(word2id,p1)
    print('完成！')

if __name__ == "__main__":
    # file_path
    if not os.path.exists('output/'): os.makedirs('output/')
    save_vector_file_name = "output/glove.txt"
    save_picture_file_name = "output/glove.png"
    embed_dir='output/glove_{}.pkl'.format(str(vector_size))
    vocabulary_dir='output/vocabulary.pkl'
    corpus_file_name = 'data/train_corpus/corpus.txt'
    train_model(epoches, corpus_file_name)
    vec_eval = VectorEvaluation(save_vector_file_name)
    # vec_eval.drawing_and_save_picture(save_picture_file_name)
    vec_eval.get_similar_words("加拿大")
    vec_eval.get_similar_words("男人")
    get_dict_and_embedding(model_dir=save_vector_file_name,embed_dir=embed_dir,vocabulary_dir=vocabulary_dir,embed_dim=vector_size)
