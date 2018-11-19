'''
    11/11/2018: Calculate the BM25 meta information. (N and dft)
'''

import os
import sys
sys.path.append("../")
sys.path.append("../../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
from nltk import word_tokenize

#   process for all type of questions
lst_fn=data_cfg.all_list_fn
q_list=gen_utils.read_dict_from_pkl(lst_fn)
candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

def calc_bm25_N():
    doc_id_list=gen_utils.read_dict_from_pkl(lst_fn)
    N=len(doc_id_list)
    return N

def calc_candidate_df():
    cand_df_dict={}
    for cand_word in candidate_list:
        if not cand_word in cand_df_dict:
            cand_df_dict[cand_word]=0

    for qid in q_list:
        print qid

        all_context_tokens=[]

        q_context_dict = data_utils.load_quaser_lctx_by_id(qid)
        for cid,context in q_context_dict.iteritems():
            cur_question=context['question']
            cur_question_tokens=word_tokenize(cur_question)
            all_context_tokens+=cur_question_tokens

        all_context_tokens=set(all_context_tokens)
        for cand_word in cand_df_dict.keys():
            if cand_word in all_context_tokens:
                cand_df_dict[cand_word]+=1



    return cand_df_dict

def calc_avg_doc_len():

    all_doc_len=[]
    for qid in q_list:
        print qid

        all_context_tokens = []

        q_context_dict = data_utils.load_quaser_lctx_by_id(qid)
        doc_len=0

        for cid, context in q_context_dict.iteritems():
            cur_question = context['question']
            cur_question_tokens = word_tokenize(cur_question)
            all_context_tokens += cur_question_tokens
            doc_len=doc_len+len(cur_question_tokens)
        all_doc_len.append(doc_len)

        if len(all_doc_len)>2000:
            break

    import numpy as np

    return np.average(all_doc_len)



if __name__=="__main__":
    dst_fn=os.path.join(data_cfg.dataset_root_path,"BM25_meta.pkl")
    bm25_meta=gen_utils.read_dict_from_pkl(dst_fn)
    avg_doc_len=calc_avg_doc_len()

    bm25_meta['avg_doc_len']=avg_doc_len

    '''
    N=calc_bm25_N()
    cand_df_dict=calc_candidate_df()
    bm25_meta={'N':N,'df_dict':cand_df_dict}
    gen_utils.write_dict_to_pkl(bm25_meta,dst_fn)
    '''

    gen_utils.write_dict_to_pkl(bm25_meta, dst_fn)
    print "done."