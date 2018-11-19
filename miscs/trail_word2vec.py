'''
    A trail program to call pre-trained word embedding model
'''

import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
from gensim.models.keyedvectors import KeyedVectors
from modules.VocabEntry import *
import numpy as np

if __name__=="__main__":
    vocab=gen_utils.read_dict_from_pkl("/home/jiangl1/data/11791_data/quaser/vocab.pkl")
    print min(vocab.word2id.values()),max(vocab.word2id.values())


    word_vect = KeyedVectors.load_word2vec_format("/home/jiangl1/data/11791_data/quaser/SO_word2vec.bin", binary=True)
    id_embedding_mat=[]
    
    for id in sorted(vocab.id2word.keys()):
        word=vocab.id2word[id]
        try:
            word_vec=word_vect[word]
        except:
            print "Embedding not found: ",word
            word_vec=np.random.rand(200)

        assert id==len(id_embedding_mat)
        id_embedding_mat.append(word_vec)
        print id

    id_embedding_mat=np.asarray(id_embedding_mat)
    np.save(os.path.join(data_cfg.raw_data_root_path,"id2wordvec.npy"),id_embedding_mat)
    print "done."