### Implemented Glove and Word2vec by Pytorch
Word Embadding 作为当下 nlp 领域最流行的技巧。为了深入了解其原理，使用当下流行的深度学习框架实现斯坦福的 `Glove` 和 google 的 `word2vec`。

* `tools.py` 提供文本的预处理，降维可视化等功能。
* `huffman.py` 将词表转化为哈夫曼树。
* `word2vec` skip-gram and cbow




#### running code
```python
python glove.py

python word2vec.py
```


**参考**
* [Glove](https://nlp.stanford.edu/projects/glove/)
* [GloVe详解](http://www.fanyeong.com/2018/02/19/glove-in-detail/)
* [pytorch_word2vec](https://github.com/bamtercelboo/pytorch_word2vec)
* [
哈夫曼树的实现](https://blog.csdn.net/IT_iverson/article/details/79018505)
