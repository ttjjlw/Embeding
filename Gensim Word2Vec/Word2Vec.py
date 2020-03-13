# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/3/13
# -*- coding: utf-8 -*-


from gensim.models import word2vec,KeyedVectors
import logging,collections,pickle,os
import numpy as np

##训练word2vec模型

# 获取日志信息
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 加载分词后的文本，使用的是Text8Corpus类
# with open ('data/train_corpus/corpus.txt','rb') as f:
#     lines=f.readlines()
#     corpus=[]
#     for line in lines:
#         corpus.extend(line.decode('utf-8').strip().split(' '))
# dic = collections.Counter(corpus).most_common()
def train(train_corpus,model_dir):
    sentences = word2vec.Text8Corpus(train_corpus)

    # 训练模型，部分参数如下
    # max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。
    #trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）。
    model = word2vec.Word2Vec(sentences, #对于大语料建议使用BrownCorpus,Text8Corpus或lineSentence构建
                              size=100, #size: 词向量的维度，默认值是100
                              alpha=0.025,#alpha： 是初始的学习速率默认值0.025，在训练过程中会线性地递减到min_alpha。
                              hs=0,#hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
                              min_count=1,#min_count:：可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
                              window=3, #window：即词向量上下文最大距离，skip-gram和cbow算法是基于滑动窗口来做预测。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。对于一般的语料这个值推荐在[5,10]之间。
                              sample=1e-3,#sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
                              seed=1,#seed：用于随机数发生器默认为1。与初始化词向量有关。
                              workers=3,#workers：用于控制训练的并行数默认为3。
                              min_alpha=0.0001,#min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每    轮的迭代步长可以由iter，alpha， min_alpha一起得出。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
                              sg=0,#0是CBOW,1是skipGram，默认为0
                              negative=5,#negative:如果大于零，则会采用negativesampling，用于设置多少个noise words（一般是5-20）。
                              iter=5,#随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
                              sorted_vocab=1,#如果为1（默认），则在分配word index 的时候会先对单词基于频率降序排序。
                              batch_words=10000 #每一批的传递给线程的单词的数量，默认为10000。

                              )


    # 保留模型，方便重用
    model.save(u'word2vec.model')
    #按词频由大到小写入word及其embedding
    model.wv.save_word2vec_format(model_dir, binary=False)

def load_pretrain_model(model_dir):
    '''
    加载word2vec预训练word embedding文件
    Args:
        model_dir: word embedding文件保存路径
    '''
    model = KeyedVectors.load_word2vec_format(model_dir)
    print('similarity(不错，优秀) = {}'.format(model.similarity("不错", "优秀")))
    print('similarity(不错，糟糕) = {}'.format(model.similarity("不错", "糟糕")))
    most_sim = model.most_similar("不错", topn=10)
    print('The top10 of 不错: {}'.format(most_sim))
    words = model.vocab
    # print(1)
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
    print('embeddings的shape为：({}*{}),{}'.format(index,embed_dim,embeddings.shape))
    with open(embed_dir,'wb') as p:
        pickle.dump(embeddings,p)
    with open(vocabulary_dir,'wb') as p1:
        pickle.dump(word2id,p1)
    print('完成！')

if __name__ == '__main__':
    train_corpus='data/train_corpus/corpus.txt'
    if not os.path.exist('output/'):os.mkdirs('output/')
    model_dir="output/Vector.txt"
    embed_dir='output/embed.pkl'
    vocabulary_dir='output/vocabulary.pkl'
    embed_dim=100
    train(train_corpus=train_corpus,model_dir=model_dir)
    get_dict_and_embedding(model_dir=model_dir,embed_dir=embed_dir,vocabulary_dir=vocabulary_dir,embed_dim=embed_dim)
    # load_pretrain_model(model_dir)