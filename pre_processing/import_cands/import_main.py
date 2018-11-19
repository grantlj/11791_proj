'''
    Import the candidate answer list
'''

import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils

org_list_fn=os.path.join(data_cfg.raw_data_root_path,"candidates.txt")
assert os.path.isfile(org_list_fn)

def read_txt_lines(fn):
    with open(org_list_fn,"r") as f:
        all_lines=f.readlines()
    all_lines=[x.replace("\n","") for x in all_lines]
    all_lines=[x.replace("\r","") for x in all_lines]
    return all_lines

def write_txt_lines(fn,all_lines):
    all_lines=[x+"\n" for x in all_lines]
    with open(fn,"w") as f:
        f.writelines(all_lines)
    return

if __name__=="__main__":
    all_cands=read_txt_lines(org_list_fn)
    write_txt_lines(data_cfg.cand_lst_fn,all_cands)
    print "done."