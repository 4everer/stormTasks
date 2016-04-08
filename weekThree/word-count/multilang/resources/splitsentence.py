# import storm
#
# class SplitSentenceBolt(storm.BasicBolt):
#     def process(self, tup):
#         words = tup.values[0].split(" ")
#         for word in words:
#           storm.emit([word])
#
# SplitSentenceBolt().run()

# encoding=utf-8
import storm
import re
import jieba
jieba.load_userdict("userdict.txt")

class SplitSentenceBolt(storm.BasicBolt):
    def process(self, tup):
        sentence = tup.values[0]
        sentence = re.sub(r"[,.;!\?]", "", sentence) # get rid of punctuation and num
        words = jieba.cut(sentence, cut_all=True)

        for word in words:
            storm.emit([word])

SplitSentenceBolt().run()