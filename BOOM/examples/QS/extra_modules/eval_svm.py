from boom.modules import Module
from multiprocessing import Pool
from nltk import word_tokenize
from nltk.corpus import stopwords
import utils.gen_utils as gen_utils
import numpy as np

stop_words = set(stopwords.words('english'))
candidate_list=gen_utils.read_lines_from_text_file("data/candidate.lst")
model=gen_utils.read_dict_from_pkl("models/MF-e.MF-i.bm25_scores.indri_scores/train/linear_svm/model.pkl")

def multi_process_helper(args):
    q_and_context_list=args[0]
    ret_list=[]

    print "In prediction: ", len(q_and_context_list), type(q_and_context_list)
    id = 0

    for q_and_context in q_and_context_list:
        id += 1
        print "In prediction, ind ", id, "/", len(q_and_context_list)

        feat=np.asarray(q_and_context['final_feat'])
        feat = np.expand_dims(feat, axis=0)
        pred_label = candidate_list[model.predict(feat)[0]]

        final_meta={}
        final_meta['pred']=pred_label
        final_meta['q_meta']=q_and_context['question']
        ret_list.append(final_meta)

    return ret_list



class eval_svm(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(eval_svm, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

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
