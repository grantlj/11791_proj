from boom.modules import Module
from multiprocessing import Pool
from nltk import word_tokenize
from nltk.corpus import stopwords
import utils.gen_utils as gen_utils
import math


indri_meta=gen_utils.read_dict_from_pkl("data/Indri_meta.pkl")

stop_words = set(stopwords.words('english'))
candidate_list = gen_utils.read_lines_from_text_file("data/candidate.lst")

lam=indri_meta['lambda']
mu=indri_meta['mu']
C=indri_meta['C']
ctf_dict=indri_meta['ctf']


def get_token_dict(tokens):
    ret_dict={}
    total_token=0
    for token in tokens:
        if not token in ret_dict:
            ret_dict[token]=0
        ret_dict[token]+=1
        total_token+=1

    return ret_dict,total_token

def multi_process_helper(args):
    q_and_context_list = args[0]
    print "In extracting Indri: ", len(q_and_context_list), type(q_and_context_list)
    ret_list = []

    id=0
    for q_and_context in q_and_context_list:

        id += 1
        print "In extracting Indri, ind ", id, "/", len(q_and_context_list)

        q_context_dict=q_and_context['context']
        all_q_tokens = []
        for qid, q_context in q_context_dict.iteritems():
            q_context_text = q_context['question']
            tokens = word_tokenize(q_context_text)
            all_q_tokens += tokens

        token_dict, doclen = get_token_dict(all_q_tokens)
        word_indri_score = {}

        for cand_word in candidate_list:
            p_mle_tc = ctf_dict[cand_word] / float(C)
            if not cand_word in token_dict:
                tf = 0.5
            else:
                tf = token_dict[cand_word]
            p_score = (1 - lam) * float(tf + mu * p_mle_tc) / float(doclen + mu) + lam * p_mle_tc
            word_indri_score[cand_word] = p_score

        q_and_context['indri_scores']=word_indri_score
        ret_list.append(q_and_context)

    return ret_list

class ext_indri_feat(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(ext_indri_feat, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf,
                                           **kwargs)

        #   number of the processes...
        self.processes = module_conf['processes'] if 'processes' in module_conf else 1
        self.pool = Pool(processes=self.processes)

    ## Override the cleanup function to make sure close the process pool.
    def cleanup(self):
        self.pool.close()
        self.pool.join()

    def process(self, job, data):
        #   get the question and context list, each item in the list is a dictionary
        q_and_context_list = data

        N = len(q_and_context_list)
        step_size = int(N / float(self.processes))
        slices = [(q_and_context_list[i:i + step_size],) for i in range(0, N, step_size)]
        tmp = self.pool.map(multi_process_helper, slices)

        result = []
        for x in tmp:
            result += x

        return result