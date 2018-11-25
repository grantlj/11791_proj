'''
 sudo pkill -KILL -u rabbitmq
 sudo service rabbitmq-server start
'''

from boom.modules import Module
from multiprocessing import Pool
from nltk import word_tokenize
from nltk.corpus import stopwords
import utils.gen_utils as gen_utils


candidate_list = gen_utils.read_lines_from_text_file("data/candidate.lst")


def most_common_in_list(tokens):
    from collections import Counter
    data = Counter(tokens)
    return data.most_common(1)[0][0]


def multi_process_helper(args):
    stop_words = set(stopwords.words('english'))
    q_and_context_list = args[0]
    ret_list = []

    print "In extracting MFE: ", len(q_and_context_list), type(q_and_context_list)

    id = 0
    for q_and_context in q_and_context_list:

        id += 1
        print "In extracting MFE, ind ", id, "/", len(q_and_context_list)
        #print q_and_context
        q_context_dict = q_and_context['context']
        all_context_tokens = []

        q_question = q_and_context['question']
        q_tokens = set(word_tokenize(q_question['question']))

        print "Finihsed token q MFE, ind ", id, "/", len(q_and_context_list)

        for qid, q_context in q_context_dict.iteritems():
            q_context_text = q_context['question']

            print "Start q context MFE, ind ", id, "/", len(q_and_context_list)

            #   LUKE: can you toeknize the text we needed as a feature using nltk as following?
            tokens = word_tokenize(q_context_text)

            #   remove the stop words
            tokens = [w for w in tokens if not w in stop_words]

            #   remove the candidates in original questions
            tokens = [w for w in tokens if not w in q_tokens]

            #   remove those words not in candidate list
            tokens = [w for w in tokens if w in candidate_list]

            all_context_tokens += tokens

        if len(all_context_tokens) != 0:
            #   obtain the one with maximum frequency
            most_common_token = most_common_in_list(all_context_tokens)
        else:
            most_common_token = None

        dst_meta = {'cand_token_list': all_context_tokens, 'most_common': most_common_token}
        q_and_context['mfe_feat'] = dst_meta

        print "Finihsed In extracting MFE, ind ", id, "/", len(q_and_context_list)
        ret_list.append(q_and_context)

    return ret_list


class ext_mfe_feat(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(ext_mfe_feat, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf,
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