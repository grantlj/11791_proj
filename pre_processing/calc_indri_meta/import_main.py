'''
    11/11/2018: Calculate the Indri meta information.
'''

import os
import sys
sys.path.append("../")
sys.path.append("../../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
from nltk import word_tokenize


lst_fn=data_cfg.all_list_fn
q_list=gen_utils.read_dict_from_pkl(lst_fn)
candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

lam=0.1
mu=1250

def get_ctf_dict():
    ret_dict={}

    for cand in candidate_list:
        ret_dict[cand]=0

    c_len=0
    for qid in q_list:


        all_context_tokens = []
        q_context_dict = data_utils.load_quaser_lctx_by_id(qid)

        for cid, context in q_context_dict.iteritems():
            cur_question = context['question']
            cur_question_tokens = word_tokenize(cur_question)
            all_context_tokens += cur_question_tokens

        token_cnt_dict={}
        for token in all_context_tokens:
            if not token in token_cnt_dict:
                token_cnt_dict[token]=0

            token_cnt_dict[token]+=1
        c_len+=len(all_context_tokens)
        for cand in candidate_list:
            if cand in token_cnt_dict:
                ret_dict[cand]+=token_cnt_dict[cand]
                print cand,ret_dict[cand]

        if qid>=5000:
            break

        print qid

    return ret_dict,c_len

if __name__=="__main__":
    indri_meta={'lambda':lam,'mu':mu}
    dst_fn = os.path.join(data_cfg.dataset_root_path, "Indri_meta.pkl")
    ctf_dict,c_len=get_ctf_dict()
    indri_meta['ctf']=ctf_dict
    indri_meta['C']=c_len
    gen_utils.write_dict_to_pkl(indri_meta,dst_fn)
    print "done."