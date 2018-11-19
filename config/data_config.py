'''
    The dataset configurations for the Quaser-S dataset.
'''

import os
import sys
sys.path.append("../")

#   the dataset root path
dataset_root_path="/mnt/hdd/public/11791/quaser/"

#   the raw data root path
raw_data_root_path=os.path.join(dataset_root_path,"raw")

if not os.path.exists(raw_data_root_path):
    os.makedirs(raw_data_root_path)

#   the list of candidate
cand_lst_fn=os.path.join(dataset_root_path,"candidate.lst")

#   the vocab entry
vocab_fn=os.path.join(dataset_root_path,"vocab.pkl")

#   the long context root path
long_ctx_root_path=os.path.join(dataset_root_path,"long_context")
short_ctx_root_path=os.path.join(dataset_root_path,"short_context")
q_root_path=os.path.join(dataset_root_path,"question")
gt_root_path=os.path.join(dataset_root_path,"gt")

#   the data split root path
data_split_root_path=os.path.join(dataset_root_path,"splits")
if not os.path.exists(data_split_root_path):
    os.makedirs(data_split_root_path)

trn_list_fn=os.path.join(data_split_root_path,"trn.pkl")
val_list_fn=os.path.join(data_split_root_path,"dev.pkl")
tst_list_fn=os.path.join(data_split_root_path,"tst.pkl")
all_list_fn=os.path.join(data_split_root_path,"all.pkl")


#   the feature root path
feat_root_path=os.path.join(dataset_root_path,"features")
if not os.path.exists(feat_root_path):
    os.makedirs(feat_root_path)

#   the prediction root path
pred_root_path=os.path.join(dataset_root_path,"predicts")
if not os.path.exists(feat_root_path):
    os.makedirs(pred_root_path)

data_split_root_path=os.path.join(dataset_root_path)
if not os.path.exists(gt_root_path):
    os.makedirs(gt_root_path)

if not os.path.exists(long_ctx_root_path):
    os.makedirs(long_ctx_root_path)

if not os.path.exists(short_ctx_root_path):
    os.makedirs(short_ctx_root_path)

if not os.path.exists(q_root_path):
    os.makedirs(q_root_path)
