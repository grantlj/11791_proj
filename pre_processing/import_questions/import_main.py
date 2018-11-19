'''
    import the context and questions from raw data.
'''

import os
import sys
import gzip
sys.path.append("../")
sys.path.append("../../")
import json
import config.data_config as data_cfg
import utils.gen_utils as gen_utils

#   context root path
lctx_root_path="/mnt/hdd/public/11791/raw/contexts/long/"
sctx_root_path="/mnt/hdd/public/11791/raw/contexts/short/"

#   question root path
q_root_path="/mnt/hdd/public/11791/raw/questions/"

split_list=['train','test',"dev"]

global gid

gid=0

def parse_ctx(ctx_list):
    id=0
    ret_dict={}
    for ctx in ctx_list:
        ctx_conf=ctx[0]
        ctx_q=str(ctx[1])
        ctx_q=ctx_q.replace("Question : ","")
        ret_dict[id]={'conf':ctx_conf,'question':ctx_q}
        id+=1
    return ret_dict



def handle_a_split(split):

    global gid


    lctx_fn=os.path.join(lctx_root_path,split+"_contexts.json.gz")
    sctx_fn=os.path.join(sctx_root_path,split+"_contexts.json.gz")
    q_fn=os.path.join(q_root_path,split+"_questions.json.gz")
    assert os.path.isfile(lctx_fn)
    assert os.path.isfile(sctx_fn)
    assert os.path.isfile(q_fn)
    print "Handling: ",lctx_fn

    f_lctx=gzip.open(lctx_fn)
    f_sctx=gzip.open(sctx_fn)
    f_q=gzip.open(q_fn)

    for lctx,sctx,q in zip(f_lctx,f_sctx,f_q):
        #print "Handling q:",q
        lctx=json.loads(lctx);sctx=json.loads(sctx);q=json.loads(q)

        lctx_dict=parse_ctx(lctx['contexts'])
        sctx_dict=parse_ctx(sctx['contexts'])

        q_meta=q
        gt=str(q_meta['answer'])

        lctx_fn=os.path.join(data_cfg.long_ctx_root_path,str(gid)+".pkl")
        sctx_fn=os.path.join(data_cfg.short_ctx_root_path,str(gid)+".pkl")
        q_meta_fn=os.path.join(data_cfg.q_root_path,str(gid)+".pkl")
        gt_fn=os.path.join(data_cfg.gt_root_path,str(gid)+".pkl")

        gid+=1
        gen_utils.write_dict_to_pkl(lctx_dict,lctx_fn)
        gen_utils.write_dict_to_pkl(sctx_dict,sctx_fn)
        gen_utils.write_dict_to_pkl(q_meta,q_meta_fn)
        gen_utils.write_dict_to_pkl(gt,gt_fn)
        pass


    f_lctx.close()
    f_sctx.close()
    f_q.close()

    return gid

'''
if __name__=="__main__":
    for split in split_list:
        cur_gid=handle_a_split(split)
        print split,cur_gid

    print "done."
'''

if __name__=="__main__":
    trn_list=[_ for _ in xrange(0,31049)]
    test_list=[_ for _ in xrange(31049,34223)]
    val_list=[_ for _ in xrange(34223,37362)]
    all_list=trn_list+test_list+val_list

    gen_utils.write_dict_to_pkl(trn_list,data_cfg.trn_list_fn)
    gen_utils.write_dict_to_pkl(val_list,data_cfg.val_list_fn)
    gen_utils.write_dict_to_pkl(test_list,data_cfg.tst_list_fn)
    gen_utils.write_dict_to_pkl(all_list,data_cfg.all_list_fn)
    print "done."