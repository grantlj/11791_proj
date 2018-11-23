'''
   11/22/2018: Viz and compare the qualitative errors on the given data.
'''

import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils

qid_list=gen_utils.read_dict_from_pkl(data_cfg.tst_list_fn)

'''
#   all correct
#   a list of correction cfgs (intersection will be applied internally)
corr_conf_list=[("MF-e.MF-i.bm25_scores.indri_scores","linear_svm"),("MF-e.MF-i","linear_svm")]

#   a list of error cfgs (intersection will be applied internally)
erro_conf_list=[]
'''

MAX_VIZ=10

'''
#   one correct
#   a list of correction cfgs (intersection will be applied internally)
corr_conf_list=[("MF-e.MF-i.bm25_scores.indri_scores","linear_svm")]

#   a list of error cfgs (intersection will be applied internally)
erro_conf_list=[("MF-e.MF-i","linear_svm")]
'''


#   both wrong
corr_conf_list=[]
erro_conf_list=[("MF-e.MF-i.bm25_scores.indri_scores","linear_svm"),("MF-e.MF-i","linear_svm")]


def filter_id_set_on_conf(conf,cand_id_set,corr):
    ret_id_set=set()
    feat_name=conf[0];clas_name=conf[1]

    pred_root_path = os.path.join(data_cfg.pred_root_path, clas_name, feat_name)
    gt_root_path = data_cfg.gt_root_path

    for qid in cand_id_set:

        gt_fn = os.path.join(gt_root_path, str(qid) + ".pkl")
        pred_fn = os.path.join(pred_root_path, str(qid) + ".pkl")
        pred_meta = gen_utils.read_dict_from_pkl(pred_fn)
        gt = gen_utils.read_dict_from_pkl(gt_fn)
        pred_term = pred_meta['pred_term']

        if corr and gt==pred_term:
            ret_id_set.add(qid)
            continue

        if (not corr) and gt!=pred_term:
            ret_id_set.add(qid)
            continue

    return ret_id_set

def filter_id_set(conf_list,corr):
    ret_qid_set=set(qid_list)

    for conf in conf_list:
        cur_qid_set=filter_id_set_on_conf(conf,ret_qid_set,corr=corr)
        ret_qid_set=cur_qid_set

    return ret_qid_set

def viz_result(final_id_set):

    viz_cnt=0
    for id in final_id_set:
        viz_cnt+=1
        q_meta=data_utils.load_quaser_qmeta_by_id(id)
        q_lctx=data_utils.load_quaser_lctx_by_id(id)
        print viz_cnt
        print q_meta

        print ""

        if viz_cnt>MAX_VIZ:
            break
        pass

    return


if __name__=="__main__":
    #  filter the correct qid set
    print "Filter on Corr..."
    corr_qid_set=filter_id_set(corr_conf_list,corr=True)

    print "Filter on Erro..."
    #  filter the error qid set
    erro_qid_set=filter_id_set(erro_conf_list,corr=False)

    final_id_set=corr_qid_set.intersection(erro_qid_set)
    print "All instances meet requirements: ",len(final_id_set),"/",len(qid_list)

    viz_result(final_id_set)

    print "done."