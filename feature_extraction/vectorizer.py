'''
    The vectorizer utilities for different features. Convert pickle format features into real vectors in numpy format.
'''
import numpy as np
import sys
import os
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils

cand_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

def vec_bm25_func(feat_root_path,qid):
    feat_fn = os.path.join(feat_root_path, str(qid) + ".pkl")
    ret_feat = np.asarray([0.0] * len(cand_list))

    try:
        org_feat = gen_utils.read_dict_from_pkl(feat_fn)
        for key,val in org_feat:
            ret_feat[cand_list.index(key)]=val
    except:
        pass

    return ret_feat

def vec_indri_func(feat_root_path,qid):
    return vec_bm25_func(feat_root_path,qid)

def vec_mfe_func(feat_root_path,qid):
    feat_fn=os.path.join(feat_root_path,str(qid)+".pkl")

    ret_feat=np.asarray([0.0]*len(cand_list))

    try:
        org_feat = gen_utils.read_dict_from_pkl(feat_fn)
        ret_feat[cand_list.index(org_feat['most_common'])]=1

        for other_token in org_feat['cand_token_list']:
            ret_feat[cand_list.index(other_token)]=0.5
    except:
        #   feature may not exist
        pass

    return ret_feat

def vec_mfi_func(feat_root_path,qid):
    #   exactly same as the mfi feature
    return vec_mfe_func(feat_root_path,qid)
