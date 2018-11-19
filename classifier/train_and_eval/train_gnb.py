'''
     11/18/2018: Classification by training a Gaussian Naive Bayes model
'''

import os
import sys
sys.path.append("../")
sys.path.append("../../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
from sklearn.naive_bayes import MultinomialNB
import numpy as np


#   a list of training configurations (feature name only for each configuration
train_cfgs=[("MF-e.MF-i.bm25_scores.indri_scores",data_cfg.trn_list_fn,"train")]
model_type="gnb"

cand_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

def load_feat_and_labels(qid_list,cfg):
    feat_name=cfg[0]
    feat_root_path=os.path.join(data_cfg.feat_root_path,feat_name)
    ret_feats=[];ret_labels=[]
    for qid in qid_list:
        print "Loading feature: ",qid
        feat_fn=os.path.join(feat_root_path,str(qid)+".npz")
        if not os.path.exists(feat_fn):
            print "Warning: feature not exist:",qid
            continue


        gt_token=data_utils.load_quaser_gt_by_id(qid)
        gt_label=cand_list.index(gt_token)
        feat=np.load(feat_fn)['feat']

        ret_feats.append(feat)
        ret_labels.append(gt_label)

    return ret_feats,ret_labels


if __name__=="__main__":

    for cfg in train_cfgs:
        qid_list=gen_utils.read_dict_from_pkl(cfg[1])
        model_root_path = os.path.join(data_cfg.model_root_path, cfg[0], cfg[2],model_type)
        print model_root_path
        all_train_feats,all_train_labels=load_feat_and_labels(qid_list,cfg)
        print "training model, cfg: ",cfg
        model=MultinomialNB()
        model=model.fit(all_train_feats,all_train_labels)
        if not os.path.exists(model_root_path):
            os.makedirs(model_root_path)
        model_fn=os.path.join(model_root_path,"model.pkl")
        gen_utils.write_dict_to_pkl(model,model_fn)

