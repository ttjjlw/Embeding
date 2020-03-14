# Embeding分为三个部分
### 1、pytorch实现glove
### 2、tensorflow 实现word2vec(skipgram) 
### 3、gensim训练word2vec词向量并保存  
  
其中第2部分是自己实现的：
输入为：数字编码的文本，word2id和id2word
输出为：词向量  

## [详细说明，请阅读博客](https://blog.csdn.net/qq_40859560/article/details/104848972)  

### 若出现'没有**module'等错误可参考如下安装  
windows系统：win+r 输入cmd，回车

pip install jieba -i https://pypi.douban.com/simple/    
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gensim  
nvcc -V 查看cuda版本   
conda install pytorch torchvision cudatoolkit=9.0（9.0为cuda的版本号）  
torch.cuda.is_available() 

