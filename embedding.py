
import pandas as pd
import os
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import LdaMulticore
import json
import gensim
import config


def label2idx(data):
    # 加载所有类别， 获取类别的embedding， 并保存文件
    if os.path.exists('./data/label2id.json'):
        labelToIndex = json.load(open('./data/label2id.json',
                                      encoding='utf-8'))
    else:
        label = data['label']#11个类别
        label=label.unique()#将serirs去重返回ndarray
        labelToIndex = dict(zip(label, list(range(len(label)))))
        with open('./data/label2id.json', 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in labelToIndex.items()}, f)
    return labelToIndex



class Embedding():
    def __init__(self):
        self.stopWords = [x.strip() for x in open('./data/stopwords.txt', encoding='utf-8').readlines()]

    def load_data(self, path):
        data = pd.read_csv(path, sep='\t')
        data = data.fillna("")#填充
        data["text"] = data['text'].apply(lambda x: " ".join([w for w in x.split()
                                                              if w not in self.stopWords and w != '']))
        self.labelToIndex = label2idx(data)
        data['label'] = data['label'].map(self.labelToIndex)
        data['label'] = data.apply(lambda row: float(row['label']), axis=1)
        data = data[['text', 'label']]
        self.train = data['text'].tolist()

    def trainer(self):
        #tfidf
        count_vect = TfidfVectorizer(stop_words=self.stopWords,
                                     max_df=0.4,
                                     min_df=0.001,
                                     ngram_range=(1, 2))
        self.tfidf = count_vect.fit(self.train)
        #word2vec
        self.train = [sample.split() for sample in self.train]
        self.w2v = models.Word2Vec(min_count=2,
                                   window=5,
                                   size=300,
                                   sample=6e-5,
                                   alpha=0.03,
                                   min_alpha=0.0007,
                                   negative=15,
                                   workers=4,
                                   iter=30,
                                   max_vocab_size=50000)
        self.w2v.build_vocab(self.train)
        self.w2v.train(self.train,
                       total_examples=self.w2v.corpus_count,
                       epochs=15,
                       report_delay=1)
        #LDA
        self.id2word = gensim.corpora.Dictionary(self.train)
        corpus = [self.id2word.doc2bow(text) for text in self.train]
        self.LDAmodel = LdaMulticore(corpus=corpus,
                                     id2word=self.id2word,
                                     num_topics=30,
                                     workers=4,
                                     chunksize=4000,
                                     passes=7,
                                     alpha='asymmetric')

    def saver(self):
        joblib.dump(self.tfidf, './data/tfidf')
        self.w2v.wv.save_word2vec_format('./data/w2v.bin',
                                         binary=False)
        self.LDAmodel.save('./data/lda')
    def load(self):
        self.tfidf = joblib.load('./data/tfidf')
        self.w2v = models.KeyedVectors.load_word2vec_format('./data/w2v.bin', binary=False)
        self.lda = models.ldamodel.LdaModel.load('./data/lda')


if __name__ == "__main__":
    em = Embedding()
    em.load_data(config.train_data_file)
    em.trainer()
    em.saver()