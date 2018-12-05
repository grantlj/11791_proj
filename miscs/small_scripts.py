import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils

if __name__=="__main__":
    gt_text, lctx_text, sctx_text, q_meta=data_utils.load_quaser_all_by_id(100)
    print gt_text
    print lctx_text
    print sctx_text
    print q_meta
    print "done."