import os
import sys
sys.path.append("../")
import config.data_config as data_cfg
import utils.gen_utils as gen_utils
import utils.data_utils as data_utils
from nltk import word_tokenize
from nltk.corpus import stopwords
import threading
import nltk

stop_words = set(stopwords.words('english'))
vocabulary = gen_utils.read_dict_from_pkl(os.path.join(data_cfg.root_path,"vocab.pkl"))

lst_fn=data_cfg.all_list_fn

candidate_list=gen_utils.read_lines_from_text_file(data_cfg.cand_lst_fn)

def window_features(qid, win_size):
    qtokens = gen_utils.read_dict_from_pkl(os.path.join(data_cfg.root_path))
    clozeList = []
    for i in range(qtokens.size()):
        token = qtokens[i]
        if token in candidate_list:
            clozeList.append(i)
            
    window_features = [0] * vocabulary.size()

    #Trying to cut down on unnecessary loops
    for cid in clozeList:
        start_win = min(win_size,cid)
        for i in range(start_win):
            winid = clozeid - (start_win - i)
            try:
                winword = vocabulary.word2id.loopkup(qtokens[winid])
                window_features[winword] = 1
            except:
                break

        end_win = min(win_size,len(qtokens) - clozeid)
        for j in range(end_win):
            winid = clozeid + i + 1
            try:
                winword = vocabulary.word2id.lookup(qtokens[winid])
                window_features[winword] = 1
            except:
                break

    win_feat_fn=os.path.join(feat_root_path,str(qid)+".pkl")
    gen_utils.write_dict_to_pkl(window_features,win_feat_fn)

    return

if __name__ == '__main__':
    win_size = int(sys.argv[1])
    
    qa_list=gen_utils.read_dict_from_pkl(lst_fn)

    if win_size >= 60:
        feat_type = "bag_of_words"
    else:
        feat_type="window_words"

    feat_root_path=os.path.join(data_cfg.feat_root_path,feat_type)
    if not os.path.exists(feat_root_path):
        os.makedirs(feat_root_path)

    #I don't understand what this does. At all.
    thread_pool = []

    for qid in qa_list:
        window_features(qid, win_size)

    for th in thread_pool:
        th.join()

    print "done"
