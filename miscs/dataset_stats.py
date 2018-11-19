'''
    11/13/2018: The dataset stats information on the Quaser-S dataset.
'''

import os
import sys
sys.path.append("../")
sys.path.append("../../")
import config.data_config as data_cfg
import utils.data_utils as data_utils
import utils.gen_utils as gen_utils

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)
#qid_list_fn=data_cfg.trn_list_fn
#qid_list_fn=data_cfg.val_list_fn
#qid_list_fn=data_cfg.test_list_fn
qid_list_fn=data_cfg.tst_list_fn

#vocab=gen_utils.read_dict_from_pkl(data_cfg.vocab_fn)
#print "done."

def get_cand_gt_dist():


    qid_list=gen_utils.read_dict_from_pkl(qid_list_fn)
    #   calculate the candidate ground-truth distribution
    count_dict={}
    for cand in candidate_list:

        question=data_utils.load_quaser_qmeta_by_id(0)
        print "pase"

        count_dict[cand]=0
    for qid in qid_list:
        gt=data_utils.load_quaser_gt_by_id(qid)
        count_dict[gt]+=1
    for key,val in count_dict.iteritems():
        print key,val/float(sum(count_dict.values()))
    return

if __name__=="__main__":
    get_cand_gt_dist()
    print "done."