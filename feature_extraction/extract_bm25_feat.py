'''
     11/11/2018: Extract the BM25 features.
'''

import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
from nltk import word_tokenize
from nltk.corpus import stopwords
import threading
import nltk
import math
import numpy as np

#   process for all type of questions
lst_fn=data_cfg.all_list_fn

#   the name of the feature
feat_type="bm25_scores"

MAX_TH=32

#   create the feature root path
feat_root_path=os.path.join(data_cfg.feat_root_path,feat_type)
if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)
bm25_meta=gen_utils.read_dict_from_pkl(os.path.join(data_cfg.dataset_root_path,"BM25_meta.pkl"))
N=bm25_meta['N']
avg_doclen=bm25_meta['avg_doc_len']
df_dict=bm25_meta['df_dict']

#   BM25 parameters
k1=1.2;b=0.75;k3=500
print "done."

def get_token_dict(tokens):
    ret_dict={}
    total_token=0
    for token in tokens:
        if not token in ret_dict:
            ret_dict[token]=0
        ret_dict[token]+=1
        total_token+=1

    return ret_dict,total_token


def extract_feat_on_qid(qid):
    print qid
    dst_feat_fn = os.path.join(feat_root_path, str(qid) + ".pkl")

    q_context_dict = data_utils.load_quaser_lctx_by_id(qid)
    all_q_tokens=[]
    for qid, q_context in q_context_dict.iteritems():
        q_context_text = q_context['question']
        tokens = word_tokenize(q_context_text)
        all_q_tokens+=tokens

    token_dict,doclen=get_token_dict(all_q_tokens)

    word_bm25_score={}
    for cand_word in candidate_list:

        try:
            df=df_dict[cand_word]
        except:
            df=0

        try:
            tf=token_dict[cand_word]
        except:
            tf=0

        rsj_weight=math.log((N-df+0.5)/float(df+0.5))
        tf_weight=tf/float(tf+k1*((1-b)+b*(doclen/float(avg_doclen))))
        user_weight=(k3+1)*1/float((k3+1))

        all_score=rsj_weight*tf_weight*user_weight
        word_bm25_score[cand_word]=all_score
        if all_score!=0:
            print cand_word, qid
        pass

    gen_utils.write_dict_to_pkl(word_bm25_score,dst_feat_fn)
    return

if __name__=="__main__":
    qa_list = gen_utils.read_dict_from_pkl(lst_fn)

    #   use multi-thread for fast processing
    thread_pool = []

    for qid in qa_list:


        th = threading.Thread(target=extract_feat_on_qid, args=(qid,))
        th.start()
        thread_pool.append(th)

        while len(threading.enumerate()) >= MAX_TH:
            pass


        #extract_feat_on_qid(qid)

    for th in thread_pool:
        th.join()


    print "done."