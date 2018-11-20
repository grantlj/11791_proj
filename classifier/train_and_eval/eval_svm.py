'''
    Evaluation the gaussian SVM model.
'''

import os
import sys
sys.path.append("../")
sys.path.append("../../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import numpy as np

cand_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

#   feature type, classifier type
train_eval_cfgs=[("MF-e.MF-i.bm25_scores.indri_scores",data_cfg.trn_list_fn,"train",data_cfg.all_list_fn)]
model_type="linear_svm"

def pred_a_cfg(cfg):


    feat_type=cfg[0];trn_split=cfg[2];eval_list=cfg[-1]

    feat_root_path = os.path.join(data_cfg.feat_root_path, feat_type)
    pred_root_path = os.path.join(data_cfg.pred_root_path,model_type,feat_type)
    if not os.path.exists(pred_root_path):
        os.makedirs(pred_root_path)

    model_root_path = os.path.join(data_cfg.model_root_path,feat_type,trn_split,model_type)
    model_fn=os.path.join(model_root_path,"model.pkl")
    model=gen_utils.read_dict_from_pkl(model_fn)


    qid_list=gen_utils.read_dict_from_pkl(eval_list)
    for qid in qid_list:
        print "Loading feature: ", qid
        feat_fn = os.path.join(feat_root_path, str(qid) + ".npz")
        if not os.path.exists(feat_fn):
            print "Warning: feature not exist:", qid
            continue

        feat=np.load(feat_fn)['feat']
        feat=np.expand_dims(feat, axis=0)
        pred_label = cand_list[model.predict(feat)[0]]
        all_pred_score = model.decision_function(feat)[0].tolist()

        q_pred = {'all_pred_probs': all_pred_score, 'pred_term': pred_label}
        pred_fn=os.path.join(pred_root_path,str(qid)+".pkl")
        gen_utils.write_dict_to_pkl(q_pred,pred_fn)


        pass

    return

if __name__=="__main__":
    for cfg in train_eval_cfgs:
        pred_a_cfg(cfg)

    print "done."