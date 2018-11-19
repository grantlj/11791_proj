'''
    11/11/2018: Extract the word-distance features.

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

MAX_TH=60

stop_words = set(stopwords.words('english'))

#   process for all type of questions
lst_fn=data_cfg.all_list_fn

#   the name of the feature
feat_type="word_dist"

#   create the feature root path
feat_root_path=os.path.join(data_cfg.feat_root_path,feat_type)
if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)


def calc_avg_dist(cand_token,token_list):
    all_dist=0.0
    for dst_token in token_list:
        all_dist+=nltk.masi_distance(cand_token,set(list(dst_token)))*1.0

    try:
        return all_dist/float(len(token_list))
    except:
        return 0


#   the candidate list (all predictions should within this list)
def extract_feat_on_qid(qid):
    print qid
    q_context_dict=data_utils.load_quaser_lctx_by_id(qid)

    #   load the query sentences, so that we can exclude it accordingly
    q_query=data_utils.load_quaser_qmeta_by_id(qid)
    q_question=q_query['question']
    q_tokens=set(word_tokenize(q_question))

    dst_feat_fn=os.path.join(feat_root_path,str(qid)+".pkl")

    all_context_tokens=[]

    for qid,q_context in q_context_dict.iteritems():
        q_context_text=q_context['question']
        #   LUKE: can you toeknize the text we needed as a feature using nltk as following?
        tokens = word_tokenize(q_context_text)

        #   remove the stop words
        tokens = [w for w in tokens if not w in stop_words]

        #   remove the candidates in original questions
        tokens = [w for w in tokens if (not w in q_tokens)]

        all_context_tokens+=tokens

    all_context_tokens=set(all_context_tokens)
    token_avg_dist_dict={}
    for cand_token in candidate_list:
        #print cand_token
        dist=calc_avg_dist(set(list(cand_token)),all_context_tokens)
        token_avg_dist_dict[cand_token]=dist
    gen_utils.write_dict_to_pkl(token_avg_dist_dict,dst_feat_fn)

    return

if __name__=="__main__":
    qa_list=gen_utils.read_dict_from_pkl(lst_fn)

    #   use multi-thread for fast processing
    thread_pool=[]

    #   extract feature for a specific question/answer context
    for qid in qa_list:

        '''
        
        th=threading.Thread(target=extract_feat_on_qid,args=(qid,))
        th.start()
        thread_pool.append(th)
        
        while len(threading.enumerate())>=MAX_TH:
            pass

        '''

        extract_feat_on_qid(qid)


    for th in thread_pool:
        th.join()

    print "done."