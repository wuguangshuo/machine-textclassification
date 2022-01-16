
import json
import jieba
import joblib
import lightgbm as lgb
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance
from embedding import Embedding
from features import (get_basic_feature, get_embedding_feature,
                      get_lda_features, get_tfidf)


class Classifier:
    def __init__(self, train_mode=False) -> None:
        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(open('./data/label2id.json', encoding='utf-8'))
        self.idx2label = {v: k for k, v in self.labelToIndex.items()}
        if train_mode:
            self.train = pd.read_csv('./data/train.csv',sep='\t').dropna().reset_index(drop=True)
            self.train=self.train[:1000]
            self.dev = pd.read_csv('./data/eval.csv',sep='\t').dropna().reset_index(drop=True)
            self.dev = self.dev[:1000]
            self.test = pd.read_csv('./data/test.csv',sep='\t').dropna().reset_index(drop=True)
            self.test = self.test[:1000]
        self.exclusive_col = ['text', 'lda', 'bow', 'label']

    def feature_engineer(self, data):
        data = get_tfidf(self.embedding.tfidf, data)
        data = get_embedding_feature(data, self.embedding.w2v)
        data = get_lda_features(data, self.embedding.lda)
        data = get_basic_feature(data)
        return data

    def trainer(self):
        self.train = self.feature_engineer(self.train)
        self.dev = self.feature_engineer(self.dev)
        cols = [x for x in self.train.columns if x not in self.exclusive_col]
        X_train = self.train[cols]
        y_train = self.train['label']
        X_test = self.dev[cols]
        y_test = self.dev['label']

        mlb = MultiLabelBinarizer(sparse_output=False)
        y_train_new = []
        y_test_new = []
        for i in y_train:
            y_train_new.append([i])
        for i in y_test:
            y_test_new.append([i])

        y_train = mlb.fit_transform(y_train_new)
        y_test = mlb.transform(y_test_new)
        print('X_train: ', X_train.shape,'y_train: ', y_train.shape)
        print(mlb.classes_)
        self.clf_BR = BinaryRelevance(classifier=lgb.LGBMClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            silent=True,
            objective='binary',
            nthread=-1,
            reg_alpha=0,
            reg_lambda=1,
            device='gpu',
            missing=None),
                                      require_dense=[False, True])
        self.clf_BR.fit(X_train, y_train)
        prediction = self.clf_BR.predict(X_test)
        print(prediction)
        print(y_test)
        print(metrics.accuracy_score(y_test, prediction))

    def save(self):
        joblib.dump(self.clf_BR, './model/clf_BR')

    def load(self):
        self.model = joblib.load('./model/clf_BR')


    def predict(self, text):
        df = pd.DataFrame([[text]], columns=['text'])
        df['text'] = df['text'].apply(lambda x: " ".join(
            [w for w in jieba.cut(x) if w not in self.stopWords and w != '']))
        df = get_tfidf(self.embedding.tfidf, df)
        df = get_embedding_feature(df, self.embedding.w2v)
        df = get_lda_features(df, self.embedding.lda)
        df = get_basic_feature(df)
        cols = [x for x in df.columns if x not in self.exclusive_col]
        pred = self.model.predict(df[cols]).toarray()[0]
        print(pred)
        print(self.idx2label)
        return [self.idx2label.get(i) for i in range(len(pred)) if pred[i] > 0]

if __name__ == "__main__":
    bc = Classifier(train_mode=True)
    bc.trainer()
    bc.save()
