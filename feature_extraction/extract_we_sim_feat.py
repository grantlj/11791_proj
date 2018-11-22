'''
    11/21/2018: Extract the binnized word embedding similarity vectors for query-question, query-context, etc.
'''

import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
import threading
import nltk
import numpy as np
from scipy import spatial

#   process for all type of questions
lst_fn=data_cfg.all_list_fn

#   the name of the feature
feat_type="w_embed_sim"
MAX_TH=16

stop_words = set(stopwords.words('english'))

#   create the feature root path
feat_root_path=os.path.join(data_cfg.feat_root_path,feat_type)
if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

word_embed_mat=np.load("/home/jiangl1/data/11791_data/quaser/raw/id2wordvec.npy")
vocab=gen_utils.read_dict_from_pkl("/home/jiangl1/data/11791_data/quaser/vocab.pkl")

print "Initialize finished..."

def calc_bin_vec(cand_vec,token_list):
    bin=np.zeros(10)
    for token in token_list:
        try:
            token_vec=word_embed_mat[vocab.word2id[token]]
        except:
            continue
        sim = 1 - spatial.distance.cosine(cand_vec, token_vec)
        sim_dim = int((sim+1)/0.2-1) #(from -1 to 1, 0.2 per interval)
        bin[sim_dim]+=1

    bin=bin/float(np.linalg.norm(bin)+0.0000001)

    return bin


def extract_feat_on_qid(qid):
    print "qid: ",qid
    q_meta=data_utils.load_quaser_qmeta_by_id(qid)
    #q_lctx=data_utils.load_quaser_lctx_by_id(qid)

    q_question=q_meta['question']
    q_question_tokens = set(word_tokenize(q_question))
    q_question_tokens = [w for w in q_question_tokens if not w in stop_words]

    feat_fn=os.path.join(feat_root_path,str(qid)+".pkl")
    cand2question_dict={}
    #cand2context_dict={}

    for cand in candidate_list:
        #print cand
        #cid2bin_dict={}

        cand_vec=word_embed_mat[vocab.word2id[cand]]
        cand_question_vec=calc_bin_vec(cand_vec,q_question_tokens)
        cand2question_dict[cand]=cand_question_vec

        '''
        for ctx_id,ctx_str in q_lctx.iteritems():
            ctx_str=ctx_str['question']
            ctx_str_tokens = set(word_tokenize(ctx_str))
            ctx_str_tokens = [w for w in ctx_str_tokens if not w in stop_words]

            cur_bin=calc_bin_vec(cand_vec,ctx_str_tokens)
            cid2bin_dict[ctx_id]=cur_bin

        cand2context_dict[cand]=cid2bin_dict
        '''

    gen_utils.write_dict_to_pkl(cand2question_dict,feat_fn)

    return

if __name__ == "__main__":
    qa_list = gen_utils.read_dict_from_pkl(lst_fn)

    #   use multi-thread for fast processing
    thread_pool = []

    #   extract feature for a specific question/answer context
    for qid in qa_list:


        th=threading.Thread(target=extract_feat_on_qid,args=(qid,))
        th.start()
        thread_pool.append(th)

        while len(threading.enumerate())>=MAX_TH:
            pass


        #extract_feat_on_qid(qid)

    for th in thread_pool:
        th.join()

    print "done."
