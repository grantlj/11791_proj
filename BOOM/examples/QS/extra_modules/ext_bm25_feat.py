from boom.modules import Module
from multiprocessing import Pool
from nltk import word_tokenize
from nltk.corpus import stopwords
import utils.gen_utils as gen_utils
import math


bm25_meta=gen_utils.read_dict_from_pkl("data/BM25_meta.pkl")
N=bm25_meta['N']
avg_doclen=bm25_meta['avg_doc_len']
df_dict=bm25_meta['df_dict']
stop_words = set(stopwords.words('english'))
candidate_list = gen_utils.read_lines_from_text_file("data/candidate.lst")

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
    k1=args[1];b=args[2];k3=args[3]
    #k1 = 1.2;b = 0.75;k3 = 500

    print "In extracting BM25: ", len(q_and_context_list), type(q_and_context_list)
    ret_list = []

    id=0
    for q_and_context in q_and_context_list:

        id += 1
        print "In extracting BM25, ind ", id, "/", len(q_and_context_list)

        q_context_dict = q_and_context['context']
        all_q_tokens = []
        for qid, q_context in q_context_dict.iteritems():
            q_context_text = q_context['question']
            tokens = word_tokenize(q_context_text)
            all_q_tokens += tokens

        token_dict, doclen = get_token_dict(all_q_tokens)

        word_bm25_score = {}
        for cand_word in candidate_list:

            try:
                df = df_dict[cand_word]
            except:
                df = 0

            try:
                tf = token_dict[cand_word]
            except:
                tf = 0

            rsj_weight = math.log((N - df + 0.5) / float(df + 0.5))
            tf_weight = tf / float(0.000001+tf + k1 * ((1 - b) + b * (doclen / float(avg_doclen+0.000001))))
            user_weight = (k3 + 1) * 1 / float((k3 + 1))

            all_score = rsj_weight * tf_weight * user_weight
            word_bm25_score[cand_word] = all_score

        q_and_context['bm25_scores']=word_bm25_score
        ret_list.append(q_and_context)

    return ret_list

class ext_bm25_feat(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(ext_bm25_feat, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf,
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
        slices = [(q_and_context_list[i:i + step_size],job.params['k1'],job.params['b'],job.params['k3'],) for i in range(0, N, step_size)]
        #slices = [(q_and_context_list[i:i + step_size],) for i in
        #          range(0, N, step_size)]
        tmp = self.pool.map(multi_process_helper, slices)

        result = []
        for x in tmp:
            result += x

        return result