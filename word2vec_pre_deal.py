# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
import pickle
import collections
import numpy as np
import pandas as pd
def token2idx(token, dictionary):
    return dictionary[token]

def sent2id(seg_words, dictionary):
    return [token2idx(token, dictionary) for token in seg_words]

#读进数据，dictionary data_num
with open (r'300dim_03\dictionary03.pkl','rb') as f:
    dictionary=pickle.load(f) #长度240000-60 word2id <PAD>:0 
    print(len(dictionary))
    dictionary_reverse=pickle.load(f) # id2word
with open(r'300dim_03\word_all_num03.pkl','rb') as f:
    data_num=pickle.load(f)
print(len(data_num))     #111997581  #111997581     
print(data_num[-1])
data_index=0
def build_batch(data,batch_size,num_skip,skip_window):
    global data_index           #data_index=0，不能放在里面，因为每次生成batch，data_index都要继承前面的
    assert batch_size % num_skip==0
    assert num_skip<=skip_window*2

    train_batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    train_label=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1 #入队长度
    deque=collections.deque(maxlen=span) #创建双向队列，如deque=[1,2,3],deque.append(4),则deque=[2,3,4]
    #初始化deque,把data前三个元素，放入deque中
    for _ in range(span):
        deque.append(data_index)
        data_index+=1
    for i in range(batch_size//num_skip):
        for j in range(span):
            if j>skip_window:
                train_batch[num_skip*i+j-1]=deque[skip_window]  ##为什么是num_skip*i，num_skip表示每次i循环，train_batch,添加了几个元素
                train_label[num_skip*i+j-1,0]=deque[j]
            elif j==skip_window:
                continue
            else:
                train_batch[num_skip*i+j]=deque[skip_window]
                train_label[num_skip*i+j,0]=deque[j]
        deque.append(data[data_index])
        data_index+=1
        data_index%=len(data) #防止最后一个batch时，data_index溢出

    return train_batch,train_label

import tensorflow as tf
#开始训练
learning_rate_base=1.2    #d当用Adam做优化器，用学习率指数变化就会报错
vocabulary_size=len(dictionary)+1           #防止总共不同的词数少于设置总词长，导致后面的embedding个数超过字典的总长，导致报keyerror
batch_size=512
embedding_size=300
neg_samples=75
num_skip=2
skip_window=1
top_k=8
#验证数据
valid_size=5
valid_window=120 #验证单词只从频率最高的120个单词中选出
#字典中没有0和1，所以从2开始选
valid_example=np.random.choice(range(1,valid_window+1),valid_size,replace=False)#repalce=False表示不重复，不重复从0-119选出20个 type array

#搭建TensorFlow框架
class Word2vec(object):
    def __init__(self,mode='train'):
        self.mode=mode
    def build_graph(self):

        self.train_x=tf.placeholder(tf.int32,[None],name='train_x')
        self.train_y=tf.placeholder(tf.int32,[None,1],name='train_y')
        if self.mode=='train':
            self.valid_data=tf.constant(valid_example,tf.int32,name='valid_data')
        else:
            self.valid_data= tf.placeholder(tf.int32, shape=None)
        self.embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1,1),name='embeddings')
        # embeddinga=tf.Variable(tf.random_normal([vocabulary_size,embedding_size]))
        # 编码的时候要注意，频率高的词用小数字编码，文本中词的编码是否一定要从零开始，
        #如果不从零编码，vocabulary_size应该等于最大的那个数字编码+1，而不应是字典的长度
        self.embed=tf.nn.embedding_lookup(self.embeddings,self.train_x)
        self.nce_weight=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],
                                                   stddev=1.0/np.sqrt(embedding_size)),name='nce_weight')
        self.nce_bias=tf.Variable(tf.zeros([vocabulary_size]),name='nce_bias')
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), axis=1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm  # 除以其L2范数后得到标准化后的normalized_embeddings
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                              self.valid_data)  # 如果输入的是64，那么对应的embedding是normalized_embeddings第64行的vector
        self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True) #shape（20,2000） # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
        print('graph build successfully!')
        if self.mode=='train':
            self.nce_loss = tf.reduce_mean(tf.nn.nce_loss(inputs=self.embed, weights=self.nce_weight, biases=self.nce_bias, num_sampled=neg_samples, labels=self.train_y,
                               num_classes=vocabulary_size))
            self.global_step=tf.Variable(0,trainable=False)
            self.learning_rate = tf.train.exponential_decay(learning_rate_base, self.global_step,len(data_num)//batch_size, 0.96)
            # self.train_ = tf.train.AdamOptimizer(self.learning_rate).minimize(self.nce_loss, global_step=self.global_step)
            self.train_ = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.nce_loss,global_step=self.global_step)
            self.init=tf.global_variables_initializer()
    def save(self,sess,save_path):
        saver=tf.train.Saver()                      # 1/save model
        self.saved_path=saver.save(sess,save_path,global_step=self.global_step)
        print('{}-save model finished!'.format(self.saved_path))
    def restore(self,sess,load_path):
        restored=tf.train.Saver()
        restored.restore(sess,load_path)
        print('{}-restored Finished!'.format(load_path))
    def training(self,sess,save_path=None,load_path=None):
        sess.run(self.init)
        if load_path:
            self.restore(sess,load_path)
        for steps in range(10000001):
            train_batch,train_label=build_batch(data_num,batch_size,num_skip,skip_window)
            feed_dict={self.train_x:train_batch,self.train_y:train_label}
            _,loss,learn_rate=sess.run([self.train_,self.nce_loss,self.learning_rate],feed_dict=feed_dict)
            #每隔500次打印一次
            if steps%10000==0:
                print('learning_rate',learn_rate)
                print('loss:',loss)
            # if steps%100000==0:
            #     print(normalized_embeddings.eval()[0:4,:])
            #计算相似性
            # 每10000次，验证单词与全部单词的相似度，并将与每个验证单词最相似的8个找出来。
            if steps % 10000== 0:
                sim = self.similarity.eval()
                for i in range(valid_size):
                    valid_word = dictionary_reverse[valid_example[i]]  # 得到验证单词
                    top_k = 5
                    nearest = (-sim[i, :]).argsort()[0:top_k+1]  # 每一个valid_example相似度最高的top-k个单词,除了自己
                    log_str = "Nearest to %s:" % valid_word
                    for index in nearest:
                        close_word_similarity = sim[i, index]
                        close_word = dictionary_reverse[index]
                        log_str = "%s %s(%s)," % (log_str, close_word, close_word_similarity)
                    print(log_str)
        # final_embedding = normalized_embeddings.eval()
            if steps%1000000==0 and steps!=0:
                self.save(sess,save_path=save_path)
    def inference(self,sess,load_path,is_save_vector=False):
        self.restore(sess,load_path=load_path)
        # 保存词向量
        if is_save_vector:
            embed =self.normalized_embeddings.eval()
            with open(r'final_set\embeddings300_3000001.pkl','wb') as f:
                pickle.dump(embed,f)
            print('成功保存词向量！')
        while 1:
            word = input('请输入：')
            print(word)
            if word in ['退出', 'withdraw']:
                break
            if word not in dictionary:
                print('none')                     #用return不会打印啊
            value_int = dictionary[word]
            value_int = np.array([value_int])

            sim, word_emberdding = sess.run([self.similarity, self.valid_embeddings], feed_dict={self.valid_data: value_int})
            sim_sort = (-sim[0, :]).argsort()  # index从大到小排序，index对应dictionary_reverse字典
            nearest = sim_sort[1:top_k + 1]  # 前top_k个,不包括自己
            log_str = "Nearest to %s:" % (word)
            for index in nearest:
                close_word_similarity = sim[0, index]
                close_word = dictionary_reverse[index]
                log_str = "%s: %s(%s)," % (log_str, close_word, close_word_similarity)
            print(log_str)

    def inference_batch(self,sess,load_path):
        self.restore(sess, load_path=load_path)
        for class_num in range(19):
            with open(r'key_words_summary\class{}_keywords.txt'.format(str(class_num + 1)), 'r') as f:
                word_str=f.read()
                word_lis=eval(word_str)
            id_lis=sent2id(word_lis,dictionary)
            if class_num==1:
                print('id_lis:',id_lis,len(id_lis))
            id_lis=np.array(id_lis)
            sim= sess.run(self.similarity,feed_dict={self.valid_data: id_lis})

            for i in range(len(id_lis)):
                valid_word = word_lis[i]  # 得到验证单词
                top_k = 2
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]  # 每一个valid_example相似度最高的top-k个单词,除了自己
                print('验证：',valid_word,dictionary_reverse[(-sim[i, :]).argsort()[0]])
                # log_str = "Nearest to %s:" % valid_word
                for index in nearest:
                    # close_word_similarity = sim[i, index]
                    close_word = dictionary_reverse[index]
                    word_lis.append(close_word)   #把关键词的相近词也加进关键词里
            print('length:',len(word_lis))
            with open(r'key_words_summary_sim\class{}_keywords.txt'.format(str(class_num + 1)), 'w') as f:
                f.write(str(word_lis))
                        # log_str = "%s %s(%s)," % (log_str, close_word, close_word_similarity)
                    # print(log_str)

    def inference_one(self,sess,load_path,is_save_vector=False):
        self.restore(sess,load_path=load_path)
        # 保存词向量
        if is_save_vector:
            embed =self.normalized_embeddings.eval()
            with open(r'final_set\embeddings_100001.pkl','wb') as f:
                pickle.dump(embed,f)
            print('成功保存词向量！')
        df=pd.read_csv('donothappen_words3.csv')
        unhappen_word=df['dif_word']
        sub_words=[]
        similar=[]
        with open('train_set_filter_dictionary.pkl', 'rb') as f:
            train_set_dictionary=pickle.load(f)
        for word in unhappen_word:
            word=str(word)
            if word not in dictionary:
                sub_words.append('none')                     #用return不会打印啊
                continue
            value_int = dictionary[word]
            value_int = np.array([value_int])

            sim, word_emberdding = sess.run([self.similarity, self.valid_embeddings], feed_dict={self.valid_data: value_int})
            sim_sort = (-sim[0, :]).argsort()  # index从大到小排序，index对应dictionary_reverse字典
            nearest = sim_sort[1:top_k + 1]  # 前top_k个,不包括自己

            for index in nearest:
                close_word_similarity = sim[0, index]
                close_word = dictionary_reverse[index]
                if close_word in train_set_dictionary:
                    sub_words.append(close_word)
                    similar.append(close_word_similarity)
                    break
        df['sub_words']=pd.Series(sub_words)
        df['sim']=pd.Series(similar)
        df.to_csv('sub_words_sim.csv')
        print('wc!')
#测试图是否搭建成功

# tf.reset_default_graph()
# model=Word2vec()
# model.build_graph()

#从头训练
# tf.reset_default_graph()
# with tf.Session() as sess:
#     model=Word2vec(mode='train')
#     model.build_graph()
#     model.training(sess=sess,save_path='word2vec_pre_model\save_net.ckpt',load_path=None)
#加载模型继续训练

# tf.reset_default_graph()
# with tf.Session() as sess:
#     model=Word2vec(mode='train')
#     model.build_graph()
#     model.training(sess=sess,save_path=r'300dim_03\word2vec_pre_model\save_net.ckpt',load_path=None)
#预测
tf.reset_default_graph()
with tf.Session() as sess:
    model = Word2vec(mode='inference')
    model.build_graph()
    model.inference(sess=sess,load_path=r'300dim_03\word2vec_pre_model\save_net.ckpt-3000001',is_save_vector=True)

