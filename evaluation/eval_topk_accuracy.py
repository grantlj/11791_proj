'''
    Evaluate the top-k accuracy on different feature/classifier combinations.
'''


import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils


feat_class_list=[('MF-e', "linear_svm", data_cfg.tst_list_fn),
                 ('MF-i', "linear_svm", data_cfg.tst_list_fn),
                 ('MF-e.MF-i', "linear_svm", data_cfg.tst_list_fn),
                 ("bm25_scores.indri_scores","linear_svm",data_cfg.tst_list_fn),
                 ("MF-e.MF-i.bm25_scores.indri_scores","linear_svm",data_cfg.tst_list_fn),

                 ('late_fusion.MF-e.MF-i', "linear_svm", data_cfg.tst_list_fn),
                 ("late_fusion.bm25_scores.indri_scores","linear_svm",data_cfg.tst_list_fn),
                 ("late_fusion.MF-e.MF-i.bm25_scores.indri_scores","linear_svm",data_cfg.tst_list_fn),
                 ]

cand_terms=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

topk_list=[1,5,20,50,100,500]

def pred_k_correct(all_probs,gt,k):
    import numpy as np
    all_probs=np.asarray(all_probs)
    top_k_ind=all_probs.argsort()[-k:][::-1]

    top_k_word_set=set()
    for ind in top_k_ind:
        top_k_word_set.add(cand_terms[ind])

    return gt in top_k_word_set


def evaluate_a_conf(conf):
    feat_type=conf[0];cls_type=conf[1]
    qid_list=gen_utils.read_dict_from_pkl(conf[2])
    pred_root_path=os.path.join(data_cfg.pred_root_path,cls_type,feat_type)
    gt_root_path=data_cfg.gt_root_path

    total_ins=0

    acc_dict={}
    for k in topk_list:
        acc_dict[k]=0

    for qid in qid_list:
        total_ins+=1
        gt_fn=os.path.join(gt_root_path,str(qid)+".pkl")
        pred_fn=os.path.join(pred_root_path,str(qid)+".pkl")
        pred_meta=gen_utils.read_dict_from_pkl(pred_fn)
        gt=gen_utils.read_dict_from_pkl(gt_fn)

        for k in topk_list:
            ret=pred_k_correct(pred_meta['all_pred_probs'],gt,k)
            if ret:
                acc_dict[k]+=1
        pass

    for k in topk_list:
        acc_dict[k]=float(acc_dict[k])/float(total_ins)

    return acc_dict

if __name__=="__main__":
    for conf in feat_class_list:
        print conf
        acc_dict=evaluate_a_conf(conf)
        print "Accuracy: ",acc_dict
        print ""

    print "done."
