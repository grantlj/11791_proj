'''
    11/11/2018: Extract the Maximum Frequency-e features.

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

MAX_TH=32

stop_words = set(stopwords.words('english'))

#   process for all type of questions
lst_fn=data_cfg.all_list_fn

#   the name of the feature
feat_type="MF-e"

#   create the feature root path
feat_root_path=os.path.join(data_cfg.feat_root_path,feat_type)
if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

def most_common_in_list(tokens):
    from collections import Counter
    data = Counter(tokens)
    return data.most_common(1)[0][0]

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
        tokens = [w for w in tokens if not w in q_tokens]

        #   remove those words not in candidate list
        tokens = [w for w in tokens if w in candidate_list]

        all_context_tokens+=tokens

    if len(all_context_tokens)!=0:
        #   obtain the one with maximum frequency
        most_common_token=most_common_in_list(all_context_tokens)
    else:
        most_common_token=None
    dst_meta={'cand_token_list':all_context_tokens,'most_common':most_common_token}

    gen_utils.write_dict_to_pkl(dst_meta,dst_feat_fn)

    return

if __name__=="__main__":
    qa_list=gen_utils.read_dict_from_pkl(lst_fn)

    #   use multi-thread for fast processing
    thread_pool=[]

    #   extract feature for a specific question/answer context
    for qid in qa_list:
        th=threading.Thread(target=extract_feat_on_qid,args=(qid,))
        th.start()
        thread_pool.append(th)

        while len(threading.enumerate())>=MAX_TH:
            pass

    for th in thread_pool:
        th.join()

    print "done."