import os
import sys
sys.path.append("../")
sys.path.append("../../")
import utils.gen_utils as gen_utils
import config.data_config as data_cfg
import utils.data_utils as data_utils
import nltk
nltk.download('stopwords') #This needs to be run the first time
from nltk.corpus import stopwords
from nltk import word_tokenize

stop_words = set(stopwords.words('english'))

lst_fn = data_cfg.all_list_fn

class Tokenizer:

    @classmethod
    def __init__(self):
        pass

    @classmethod
    def getTokens(self, text):
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if not w in stop_words]
        return tokens

    @classmethod
    def write_tokens(self, text, fn):
        gen_utils.write_dict_to_pkl(text, fn)

    @classmethod
    def read_data(self, qid):
        return data_utils.load_quaser_all_by_id(qid)

if __name__ == '__main__':
    tokenizer = Tokenizer()
    qa_list = gen_utils.read_dict_from_pkl(lst_fn)
    for id in qa_list:
        tokenizer.write_tokens(tokenizer.getTokens(tokenizer.read_data(id)), "tokens-"+id+".pkl")
