from boom.modules import Module
from multiprocessing import Pool
from nltk import word_tokenize
from nltk.corpus import stopwords
import utils.gen_utils as gen_utils
import numpy as np

stop_words = set(stopwords.words('english'))
cand_list=gen_utils.read_lines_from_text_file("data/candidate.lst")


def vec_bm25_func(q_meta):
    ret_feat = np.asarray([0.0] * len(cand_list))

    try:
        org_feat = q_meta['bm25_scores']
        for key,val in org_feat:
            ret_feat[cand_list.index(key)]=val
    except:
        pass

    return ret_feat

def vec_indri_func(q_meta):
    ret_feat = np.asarray([0.0] * len(cand_list))

    try:
        org_feat = q_meta['indri_scores']
        for key, val in org_feat:
            ret_feat[cand_list.index(key)] = val
    except:
        pass

    return ret_feat

def vec_mfe_func(q_meta):

    ret_feat=np.asarray([0.0]*len(cand_list))

    try:
        org_feat = q_meta['mfe_feat']
        ret_feat[cand_list.index(org_feat['most_common'])]=1

        for other_token in org_feat['cand_token_list']:
            ret_feat[cand_list.index(other_token)]=0.5
    except:
        #   feature may not exist
        pass

    return ret_feat

def vec_mfi_func(q_meta):
    #   exactly same as the mfi feature
    ret_feat = np.asarray([0.0] * len(cand_list))

    try:
        org_feat = q_meta['mfi_feat']
        ret_feat[cand_list.index(org_feat['most_common'])] = 1

        for other_token in org_feat['cand_token_list']:
            ret_feat[cand_list.index(other_token)] = 0.5
    except:
        #   feature may not exist
        pass

    return ret_feat

def early_fusion(all_feat_vec_list):
    final_feat = None
    for cur_feat in all_feat_vec_list:
        if final_feat is None:
            final_feat = cur_feat
        else:
            final_feat = np.concatenate((final_feat, cur_feat), axis=0)
    return final_feat

def multi_process_helper(args):
    q_and_context_list=args[0]
    ret_list=[]

    print "In vectorize: ", len(q_and_context_list), type(q_and_context_list)

    id = 0
    for q_and_context in q_and_context_list:

        id += 1
        print "In vectorize, ind ", id, "/", len(q_and_context_list)

        mfe_vec=vec_mfe_func(q_and_context)
        mfi_vec=vec_mfi_func(q_and_context)
        indri_vec=vec_indri_func(q_and_context)
        bm25_vec=vec_bm25_func(q_and_context)

        final_feat=early_fusion([mfe_vec,mfi_vec,bm25_vec,indri_vec])

        q_and_context['final_feat']=final_feat.tolist()

        ret_list.append(q_and_context)

    return ret_list

class vectorize(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(vectorize, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        #   number of the processes...
        self.processes = module_conf['processes'] if 'processes' in module_conf else 1
        self.pool = Pool(processes=self.processes)

    ## Override the cleanup function to make sure close the process pool.
    def cleanup(self):
        self.pool.close()
        self.pool.join()

    def process(self, job, data):

        #   get the question and context list, each item in the list is a dictionary
        q_and_context_list=data

        N=len(q_and_context_list)
        step_size = int(N / float(self.processes))
        slices = [(q_and_context_list[i:i + step_size],) for i in range(0, N, step_size)]
        tmp = self.pool.map(multi_process_helper, slices)

        result = []
        for x in tmp:
            result += x

        return result
