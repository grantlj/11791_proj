import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils

trn_lst=gen_utils.read_dict_from_pkl(data_cfg.trn_list_fn)
print "Initialize finished..."
ins_lst=trn_lst[0:2]

if __name__=="__main__":

    all_ins=[]
    for ins_id in ins_lst:
        print ins_id
        question=data_utils.load_quaser_qmeta_by_id(ins_id)
        context=data_utils.load_quaser_lctx_by_id(ins_id)
        q_and_context={'question':question,'context':context}
        all_ins.append(q_and_context)

    gen_utils.write_dict_to_json(all_ins,"/home/jiangl1/data/11791_data/jiang_codes/BOOM/examples/QS/test_data.json")
    print "done."