# -*- coding: UTF-8 -*-
from __future__ import print_function

import storm
import pickle


class VectorizePredictBolt(storm.BasicBolt):

    def __init__(self):
        super(storm.BasicBolt, self).__init__()

        self.count_vect = None
        self.tfidf_transformer = None
        self.target_names = None
        self.models = None
        self.xgb = None
        self._prediction = []

    def initialize(self, stormconf, context):
        with open('models.pickle', 'rb') as f:
            self.count_vect, self.tfidf_transformer, self.target_names, self.models = pickle.load(f)

    def process(self, tup):

        title, url, text_content = tup.values

        # print(title, url, type(text_content))

        for clf in self.models:
            new_counts = self.count_vect.transform([text_content])
            new_tfidf = self.tfidf_transformer.transform(new_counts)
            predicted = clf.predict(new_tfidf)
            self._prediction.append(self.target_names[predicted])

            storm.log(title + url + " is : ", self.target_names[predicted])

        emit_tup = [title, url, "|".join(self._prediction)]
        storm.log("Emit %s" % emit_tup)
        storm.emit(emit_tup)


VectorizePredictBolt().run()
