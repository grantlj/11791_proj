import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
import threading
import nltk

lst_fn=data_cfg.all_list_fn

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cad_lst.fn)

standard_vectors=gen_utils.read_dict_from_pkl(data_cfg.standard_vectors) #address
domain_vectors=gen_utils.read_dict_from_pkl(data_cfg.domain_vectors) #address

def context(qid, threshold):
    qtokens = gen_utils.read_dict_from_pkl(os.path.join(data_cfg.root_path, qid & "_tokens.pkl")) #check actual address
    conVectorsSt = []
    conVectorsDm = []

    targVectorsSt = []
    targVectorsDm = []
    
    for i in range(qtokens.size()):
        token = qtokens[i]
        
        qvecSt = standard_vectors.lookup(token)
        qvecDo = domain_vectors.lookup(token)

        if token in candidate_list:
            targVectorsSt.append(qvecSt)
            targVectorsDo.append(qvecDo)
        else:
            conVectorsSt.append(qvecSt)
            conVectorsDo.append(qvecDo)

    count = 0
    stanDistTotal = 0
    domDistTotal = 0
    for i in range(targVectorsSt.size()):
        count += 1
        targSt = targVectorsSt[i]
        targDo = targVectorsDo[i]

        for j in range(conVectorsSt.size()):
            conSt = conVectorsSt[j]
            conDo = conVectorsSt[j]
            
            stanDistTotal += vecDistance(targSt, conSt)
            domDistTotal += vecDistance(targDo, conDo)

    stanDistTotal = float(stanDistTotal)/count
    domDistTotal = float(domDistTotal)/count

    if stanDistTotal > domDistTotal:
        context_feat = domDistTotal/stanDistTotal
    else:
        context_feat = -1 * (stanDistTotal/domDistTotal)

    con_feat_fn=os.path.join(feat_root_path,str(qid)+".pkl")
    gen_utils.write_dict_to_pkl(context_feat,con_feat_fn)

    return
        
if __name__ == '__main__':
    threshold = float(sys.argv[1])

    qa_list=gen_utils.read_dict_from_pkl(lst_fn)

    feat_type="context"
    feat_root_path=os.path.join(data_cfg.feat_root_path,feat_type)
    if not os.path.exists(feat_root_path):
        os.makedirs(feat_root_path)

    thread_pool = []

    for qid in qa_list:
        context(qid, threshold)

    for th in thread_pool:
        th.join()

    print "done"
