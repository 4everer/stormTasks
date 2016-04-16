# -*- coding: UTF-8 -*-
from __future__ import print_function

import storm
import xgboost as xgb
import pickle
from pprint import pprint


class XgbPredictBolt(storm.BasicBolt):

    def __init__(self):
        super(storm.BasicBolt, self).__init__()

        self.count_vect = None
        self.tfidf_transformer = None
        self.target_names = None
        self.xgb = None
        self._prediction = None

    def initialize(self, stormconf, context):

        with open('models.pickle', 'rb') as f:
            self.count_vect, self.tfidf_transformer, self.target_names, _ = pickle.load(f)

        with open('xgb_model', 'rb') as f:
            self.xgb = pickle.load(f)

    def process(self, tup):

        title, url, text_content = tup.values

        # print(title, url, type(text_content))

        new_counts = self.count_vect.transform([text_content])
        new_tfidf = self.tfidf_transformer.transform(new_counts)
        new_dmatrix = xgb.DMatrix(new_tfidf)
        predicted = self.xgb.predict(new_dmatrix)

        self._prediction = self.target_names[predicted.argmax()]

        storm.log(title + url + " is : ", self._prediction)

        emit_tup = [title, url, self._prediction]
        storm.log("Emit %s" % emit_tup)
        storm.emit(emit_tup)


XgbPredictBolt().run()
