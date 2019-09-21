from collections import namedtuple
from functools import partial

import torch
from tqdm import tqdm

from cbow import COBW
from skip_gram import SkipGram
from tools import CorpusPreprocess, VectorEvaluation


def creat_params(input_file_path="./data/zhihu.txt",
                 output_vector_path="./data/vector.txt",
                 eval_picture_path="./data/eval_vec.png",
                 neg_num=16, epoches=10,
                 bach_size=128, windows_size=5,
                 embedding_dim=100, min_freq=5,
                 neg_model=True, use_skip_gram=True):

    mytuple = namedtuple("word2vec",
                         ["input_file_path",
                          "output_vector_path",
                          "eval_picture_path",
                          "neg_num",
                          "epoches",
                          "bach_size",
                          "windows_size",
                          "embedding_dim",
                          "min_freq",
                          "neg_model",
                          "use_skip_gram"])

    return mytuple(input_file_path, output_vector_path, eval_picture_path,
                   neg_num, epoches, bach_size, windows_size, embedding_dim,
                   min_freq, neg_model, use_skip_gram)

class Word2Vec():
    def __init__(self, args):

        self.input_file_path = args.input_file_path
        self.output_vector_path = args.output_vector_path
        self.eval_picture_path = args.eval_picture_path
        self.min_freq = args.min_freq
        self.bach_size = args.bach_size
        self.neg_num = args.neg_num
        self.epoches = args.epoches
        self.windows_size = args.windows_size
        self.embedding_dim = args.embedding_dim
        self.use_skip_gram = args.use_skip_gram
        self.neg_model = args.neg_model

        self.data_processor = CorpusPreprocess(
            self.input_file_path, self.min_freq)
        self.use_cuda = torch.cuda.is_available()
        self.build_model()

    def build_model(self):
        if not self.data_processor.vocab:
            self.data_processor.get_vocab()
        if self.use_skip_gram:
            self.model = SkipGram(self.embedding_dim,
                                  len(self.data_processor.vocab),
                                  self.neg_model)
        else:
            self.model = COBW(self.embedding_dim,
                              len(self.data_processor.vocab),
                              self.neg_model)
        if self.use_cuda:
            self.model.cuda()

    def train_model(self, word="中国"):
        print("start train !!!")
        optimizer = torch.optim.Adam(self.model.parameters())
        steps = 0
        for epoch in range(self.epoches):
            if self.use_skip_gram:
                data = self.data_processor.build_skip_gram_tain_data(
                    self.windows_size)
            else:
                data = self.data_processor.build_cbow_tain_data(
                    self.windows_size)

            batch_data_iter = self.data_processor.get_bach_data(
                data, self.bach_size)

            if not self.neg_model:
                get_batch_data_fn = self.data_processor.get_bath_huffman_tree_sample
            else:
                get_batch_data_fn = partial(
                    self.data_processor.get_bath_nagative_train_data, count=self.neg_num)
            for batch in batch_data_iter:
                batch_data = get_batch_data_fn(batch)
                pos_v = []
                pos_u = []
                neg_v = []
                neg_u = []
                for i in batch_data:
                    pos_v += i[0] * len(i[1])
                    pos_u += i[1]
                    neg_v += i[0] * len(i[2])
                    neg_u += i[2]
                input_type = torch.LongTensor
                if self.use_cuda:
                    input_type = torch.cuda.LongTensor
                if self.use_skip_gram:
                    pos_v = input_type(pos_v)
                    neg_v = input_type(neg_v)
                else:
                    pos_v = [input_type(i) for i in pos_v]
                    neg_v = [input_type(i) for i in neg_v]
                pos_u = input_type(pos_u)
                neg_u = input_type(neg_u)
                loss = self.model(pos_v, pos_u, neg_v, neg_u)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if steps % 1000 == 0:
                    print(
                        f"Epoch {epoch} steps {steps}, loss {loss.item()/len(batch)}, learn_rate:{optimizer.param_groups[0]['lr']}")

                steps += 1
        self.save_vector()
        self.evaluation_vector(self.eval_picture_path,
                               word=word, w_num=10)

    def save_vector(self):
        if hasattr(self.model, "embedding_matrix"):
            metrix = self.model.embedding_matrix.weight.data
        else:
            metrix = self.model.v_embedding_matrix.weight.data
        with open(self.output_vector_path, "w", encoding="utf-8") as f:
            if self.use_cuda:
                vector = metrix.cpu().numpy()
            else:
                vector = metrix.numpy()
            for i in tqdm(range(len(vector))):
                word = self.data_processor.idex2word[i]
                s_vec = vector[i]
                s_vec = [str(s) for s in s_vec.tolist()]
                write_line = word + " " + " ".join(s_vec)+"\n"
                f.write(write_line)
            print("Word2vec vector save complete!")

    def evaluation_vector(self, picture_path, word, w_num):
        ve = VectorEvaluation(self.output_vector_path)
        ve.drawing_and_save_picture(picture_path)
        ve.get_similar_words(word, w_num)


if __name__ == "__main__":
    # use huffman
    args = creat_params(neg_model=False, use_skip_gram=False)
    w2v = Word2Vec(args)
    w2v.train_model()
