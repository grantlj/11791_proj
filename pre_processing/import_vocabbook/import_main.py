import os
import sys
sys.path.append("../")
sys.path.append("../../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
from modules.VocabEntry import VocabEntry

lst_fn=data_cfg.all_list_fn
qa_list=gen_utils.read_dict_from_pkl(lst_fn)

gt_root_path=data_cfg.gt_root_path
lctx_root_path=data_cfg.long_ctx_root_path
sctx_root_path=data_cfg.short_ctx_root_path
q_root_path=data_cfg.q_root_path

cand_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

def get_q_text(qid):
    ret_lines=[]
    gt_text,lctx_dict,sctx_dict,q_meta=data_utils.load_quaser_all_by_id(qid)
    ret_lines.append(gt_text)
    for _,meta in lctx_dict.iteritems():
        ret_lines.append(meta['question'])

    for _, meta in sctx_dict.iteritems():
        ret_lines.append(meta['question'])

    ret_lines.append(q_meta['question'])
    ret_lines.append(q_meta['answer'])

    for i in xrange(0,len(ret_lines)):
        ret_lines[i]=str(ret_lines[i]).strip().split(' ')

    return ret_lines

if __name__=="__main__":

    '''
    all_texts=[]
    all_texts+=cand_list
    for qid in qa_list:
        q_texts=get_q_text(qid)
        all_texts+=q_texts
        print qid,"/",max(qa_list)

    vocab_entry = VocabEntry()
    vocab=VocabEntry.from_corpus(all_texts, 80000, 2, lower_case=False,whitelist=cand_list)
    gen_utils.write_dict_to_pkl(vocab,data_cfg.vocab_fn)
    '''

    vocab=gen_utils.read_dict_from_pkl(data_cfg.vocab_fn)

    print "done."