'''
     11/12/2018: Classfication in heuristic way by maximing over BM25 scores.
'''

import os
import sys
sys.path.append("../")
sys.path.append("../../")
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

feat_type="bm25_scores"
classifier_type="heuristic"

feat_root_path=os.path.join(data_cfg.feat_root_path,feat_type)
pred_root_path=os.path.join(data_cfg.pred_root_path,classifier_type,feat_type)
if not os.path.exists(pred_root_path):
    os.makedirs(pred_root_path)

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

if __name__=="__main__":
    qa_list = gen_utils.read_dict_from_pkl(lst_fn)
    for qid in qa_list:
        print "classify on qid: ",qid
        feat_fn = os.path.join(feat_root_path, str(qid) + ".pkl")
        bm25_dict = gen_utils.read_dict_from_pkl(feat_fn)

        cur_prob=[];max_prob=-1;max_label=None

        for cand_term in candidate_list:
            cur_prob.append(bm25_dict[cand_term])
            if bm25_dict[cand_term]>max_prob:
                max_prob=bm25_dict[cand_term]
                max_label=cand_term

        q_pred={'all_pred_probs':cur_prob,'pred_term':max_label}
        pred_fn=os.path.join(pred_root_path,str(qid)+".pkl")
        gen_utils.write_dict_to_pkl(q_pred,pred_fn)

    print "done."