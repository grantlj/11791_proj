'''
    Evaluate the accuracy on different feature/classifier combinations.
'''

import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils

'''
feat_class_list=[("bm25_scores","heuristic",data_cfg.trn_list_fn),
                 ("MF-e","heuristic",data_cfg.trn_list_fn),
                 ("MF-i","heuristic",data_cfg.trn_list_fn),
                 ("bm25_scores", "heuristic", data_cfg.val_list_fn),
                 ("MF-e", "heuristic", data_cfg.val_list_fn),
                 ("MF-i", "heuristic", data_cfg.val_list_fn),
                 ("bm25_scores", "heuristic", data_cfg.tst_list_fn),
                 ("MF-e", "heuristic", data_cfg.tst_list_fn),
                 ("MF-i", "heuristic", data_cfg.tst_list_fn)
                 ]
'''

'''
feat_class_list=[('indri_scores','heuristic',data_cfg.trn_list_fn),
                 ('indri_scores', 'heuristic', data_cfg.val_list_fn),
                 ('indri_scores', 'heuristic', data_cfg.tst_list_fn)]
'''

'''
feat_class_list=[('bm25_scores.indri_scores',"gnb",data_cfg.trn_list_fn),
                 ('bm25_scores.indri_scores', "gnb", data_cfg.val_list_fn),
                 ('bm25_scores.indri_scores', "gnb", data_cfg.tst_list_fn),
                 ('MF-e.MF-i', "gnb", data_cfg.trn_list_fn),
                 ('MF-e.MF-i', "gnb", data_cfg.val_list_fn),
                 ('MF-e.MF-i', "gnb", data_cfg.tst_list_fn),
                 ('MF-e.MF-i.bm25_scores.indri_scores', "gnb", data_cfg.trn_list_fn),
                 ('MF-e.MF-i.bm25_scores.indri_scores', "gnb", data_cfg.val_list_fn),
                 ('MF-e.MF-i.bm25_scores.indri_scores', "gnb", data_cfg.tst_list_fn),

                 ('bm25_scores.indri_scores',"linear_svm",data_cfg.trn_list_fn),
                 ('bm25_scores.indri_scores', "linear_svm", data_cfg.val_list_fn),
                 ('bm25_scores.indri_scores', "linear_svm", data_cfg.tst_list_fn),
                 ('MF-e.MF-i', "linear_svm", data_cfg.trn_list_fn),
                 ('MF-e.MF-i', "linear_svm", data_cfg.val_list_fn),
                 ('MF-e.MF-i', "linear_svm", data_cfg.tst_list_fn),
                 ('MF-e.MF-i.bm25_scores.indri_scores', "linear_svm", data_cfg.trn_list_fn),
                 ('MF-e.MF-i.bm25_scores.indri_scores', "linear_svm", data_cfg.val_list_fn),
                 ('MF-e.MF-i.bm25_scores.indri_scores', "linear_svm", data_cfg.tst_list_fn)

                 ]
'''

feat_class_list=[("word_dist","heuristic",data_cfg.trn_list_fn),
                 ("word_dist","heuristic",data_cfg.val_list_fn),
                 ("word_dist","heuristic",data_cfg.tst_list_fn)]

def evaluate_a_conf(conf):
    feat_type=conf[0];cls_type=conf[1]
    qid_list=gen_utils.read_dict_from_pkl(conf[2])
    pred_root_path=os.path.join(data_cfg.pred_root_path,cls_type,feat_type)
    gt_root_path=data_cfg.gt_root_path

    acc_count=0
    total_ins=0

    for qid in qid_list:
        total_ins+=1
        gt_fn=os.path.join(gt_root_path,str(qid)+".pkl")
        pred_fn=os.path.join(pred_root_path,str(qid)+".pkl")
        pred_meta=gen_utils.read_dict_from_pkl(pred_fn)
        gt=gen_utils.read_dict_from_pkl(gt_fn)
        pred_term=pred_meta['pred_term']
        if gt==pred_term:
            acc_count+=1

        pass

    return acc_count/float(total_ins)

if __name__=="__main__":
    for conf in feat_class_list:
        print conf
        acc=evaluate_a_conf(conf)
        print "Accuracy: ",acc
        print ""

    print "done."