'''
    Early fusion the features from various source and encoding them into a uniform length vector.
'''

import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import vectorizer as vectorizer
import utils.gen_utils as gen_utils
import numpy as np

#   the feature name, vectorizer function list
feat_list=[("bm25_scores",vectorizer.vec_bm25_func),("indri_scores",vectorizer.vec_indri_func),
           ("MF-e",vectorizer.vec_mfe_func),("MF-i",vectorizer.vec_mfi_func)]

feat_list=sorted(feat_list)
dst_feat_type=".".join([x[0] for x in feat_list])
print dst_feat_type

dst_feat_root_path=os.path.join(data_cfg.feat_root_path,dst_feat_type)
if not os.path.exists(dst_feat_root_path):
    os.makedirs(dst_feat_root_path)

q_list_fn=data_cfg.all_list_fn

def handle_a_particular_qid(qid):

    all_feat_vec_list=[]

    for feat_meta in feat_list:
        feat_name=feat_meta[0]
        org_feat_root_path=os.path.join(data_cfg.feat_root_path,feat_name)
        vec_func=feat_meta[1]
        feat_vec=vec_func(org_feat_root_path,qid)
        all_feat_vec_list.append(feat_vec)

    final_feat=None
    for cur_feat in all_feat_vec_list:
        if final_feat is None:
            final_feat=cur_feat
        else:
            final_feat=np.concatenate((final_feat,cur_feat),axis=0)

    dst_feat_fn=os.path.join(dst_feat_root_path,str(qid)+".npz")
    np.savez_compressed(dst_feat_fn,feat=final_feat)

    return

if __name__=="__main__":
    q_list=gen_utils.read_dict_from_pkl(q_list_fn)
    for qid in q_list:
        print qid
        handle_a_particular_qid(qid)
    print "done."