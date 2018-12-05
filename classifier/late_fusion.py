'''
    Late fusion the predictions from different classifier outputs.
'''

import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import numpy as np


#classifier_type="heuristic"
#classifier_type="linear_svm"

'''
fusion_lists=[("gnb",["MF-e.MF-i","bm25_scores.indri_scores"]),
              ("linear_svm", ["MF-e.MF-i", "bm25_scores.indri_scores"])]
'''

fusion_lists=[("gnb",["MF-e","MF-i"]),
              ("gnb", ["bm25_scores", "indri_scores"]),
              ("gnb",["MF-e","MF-i","bm25_scores", "indri_scores"]),

              ("linear_svm",["MF-e","MF-i"]),
              ("linear_svm", ["bm25_scores", "indri_scores"]),
              ("linear_svm",["MF-e","MF-i","bm25_scores", "indri_scores"])]


def convert_to_prob(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def handle_a_fusion_list(fusion_list):
    classifier_type=fusion_list[0]
    fusion_list=fusion_list[1]
    fusion_list=sorted(fusion_list)
    dst_feat_type=".".join(fusion_list)
    dst_feat_type="late_fusion."+dst_feat_type
    print dst_feat_type

    dst_pred_root_path=os.path.join(data_cfg.pred_root_path,classifier_type,dst_feat_type)
    if not os.path.exists(dst_pred_root_path):
        os.makedirs(dst_pred_root_path)

    q_list_fn=data_cfg.all_list_fn

    cand_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

    def handle_a_particular_qid(qid):
        final_pred_fn=os.path.join(dst_pred_root_path,str(qid)+".pkl")

        final_pred_probs=None

        for feat_type in fusion_list:
            cur_pred_fn=os.path.join(data_cfg.pred_root_path,classifier_type,feat_type,str(qid)+".pkl")
            cur_pred=gen_utils.read_dict_from_pkl(cur_pred_fn)
            cur_pred_probs=convert_to_prob(np.asarray(cur_pred['all_pred_probs']))

            if final_pred_probs is None:
                final_pred_probs=cur_pred_probs
            else:
                final_pred_probs=final_pred_probs+cur_pred_probs

        final_pred_probs=final_pred_probs.tolist()
        max_ind=np.argmax(final_pred_probs)
        final_pred_term=cand_list[max_ind]

        final_meta={'all_pred_probs':final_pred_probs,'pred_term':final_pred_term}
        gen_utils.write_dict_to_pkl(final_meta,final_pred_fn)


        return

    #   main of handle a fusion list
    q_list = gen_utils.read_dict_from_pkl(q_list_fn)
    for qid in q_list:
        print qid
        handle_a_particular_qid(qid)


if __name__=="__main__":
    for l in fusion_lists:
        handle_a_fusion_list(l)
    print "done."