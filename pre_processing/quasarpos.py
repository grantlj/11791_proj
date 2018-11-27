import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
from nltk.tag import stanford
import threading
import nltk

lst_fn=data_cfg.all_list_fn

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cad_lst.fn)

pos_tagger = stanford.StanfordPOSTagger(os.path.join(data_cfg.root_path,"english-bidirectional-distsim.tagger"))

def tag_pos(qid):
    qtokens = gen_utils.read_dict_from_pkl(os.path.join(data_cfg.root_path,qid & "_tokens.pkl"))
    qpos = pos_tagger.tag(qtokens)

    pos_list = []
    for q in qpos:
        tok, pos = q
        pos_list.append(pos)

    pos_fn=os.path.join(feat_root_path,str(qid)+".pkl")
    gen_utils.write_dict_to_pkl(pos_list,pos_fn)

    return

if __name__ == '__main__':

    qa_list=gen_utils.read_dict_from_pkl(lst_fn)

    pos_root_path=os.path.join(data_cfg.root_path,"pos")

    for qid in qa_list:
        tag_pos(qid)

    for th in thread_poll:
        th.join()

    print "done"
