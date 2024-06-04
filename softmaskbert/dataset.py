import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import random
import os
import sys
root_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_path)
from softmaskbert.utils import init_dir


class ServeDataSet(Dataset):
    def __init__(self, src_data, tokenizer,max_len=300): #句子的最大长度300，可调控
        self.src_data = src_data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, index):
        src_text = self.src_data[index]
        inputs = self.tokenizer.encode_plus(
            src_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=False,
            return_token_type_ids=True,
            truncation=True
        )
        hard_mask = self.tokenizer.encode_plus(
            ["<mask>"]*(len(src_text)),
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=False,
            return_token_type_ids=True,
            truncation=True
        )
        return {
            "input_ids":torch.tensor(inputs['input_ids'], dtype=torch.long),
            "input_mask":torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "mask_ids":torch.tensor(hard_mask["input_ids"], dtype=torch.long),
            "input_token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
        }

def serve_collate_fn(data):
    _data = data[:]
    data = dict()
    for i, d in enumerate(_data):
        for k,v in d.items():
            data_list = data.get(k, [])
            data_list.append(v)
            data[k] = data_list
    input_ids = data["input_ids"]
    input_mask = data["input_mask"]
    input_token_type_ids = data["input_token_type_ids"]
    mask_ids = data["mask_ids"]
    
    def merge(seqs):
        lens = [len(seq) for seq in seqs]
        
        padded_seqs =  torch.zeros(len(seqs),max(lens),dtype=seqs[0].dtype)
        for i, seq in enumerate(seqs):
            end = lens[i]
#             print(padded_seqs.dtype, seq.dtype)
            if seq.dtype == torch.int64:
                padded_seqs[i,:end] = torch.LongTensor(seq[:end])
            else:
                padded_seqs[i,:end] = torch.Tensor(seq[:end])
        return padded_seqs, torch.LongTensor(lens)
    
    input_ids, input_ids_len = merge(input_ids)
    input_mask, _ = merge(input_mask)
    input_token_type_ids,_ = merge(input_token_type_ids)
    mask_ids , _ = merge(mask_ids)
    return { "input_ids":input_ids,
            "input_ids_len":input_ids_len,
            "input_mask":input_mask,
            "input_token_type_ids":input_token_type_ids,
            "mask_ids":mask_ids
    }




        


class CustomDataset(Dataset):
    def __init__(self, src_file, trg_file, label_file, nraws, tokenizer, max_len, shuffle=False):
        self.tokenizer = tokenizer
        self.max_len = max_len 
        file_raws = 0
        with open(src_file,'r') as f:
            for _ in f:
                file_raws+=1
        self.file_raws = file_raws 
        self.nraws = nraws
        print("{} Samples".format(self.file_raws))
        self.src_file_path = src_file
        self.trg_file_path = trg_file
        self.label_file = label_file 
        self.shuffle = shuffle
    
    def __len__(self):
        return self.file_raws
    
    def get_nraw_data(self):
        self.src_data = []
        self.trg_data = []
        self.label_data = []
        for _ in range(self.nraws):
            _src_data = self.src_finput.readline().strip()
            _trg_data = self.trg_finput.readline().strip()
            _label_data = self.label_finput.readline().strip()
            if _src_data and _trg_data and _label_data:
                self.src_data.append(_src_data.split())
                self.trg_data.append(_trg_data.split())
                self.label_data.append(list(map(lambda x: int(x), _label_data.split())))
            else:
                break 
        self.current_sample_num = len(self.src_data)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.index)

    def initial(self):
        self.src_finput = open(self.src_file_path, "r")
        self.trg_finput = open(self.trg_file_path, "r")
        self.label_finput = open(self.label_file, "r")
        self.get_nraw_data()
        

    def __getitem__(self,index):
        idx = self.index[0]
        src_text = self.src_data[idx]
        trg_text = self.trg_data[idx]
        label = [1]+self.label_data[idx]+[1]
        inputs = self.tokenizer.encode_plus(
            src_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=False,
            return_token_type_ids=True
        )
        hard_mask = self.tokenizer.encode_plus(
            ["<mask>"]*(len(src_text)+2),
            None,
            add_special_tokens=False,
            max_length=self.max_len,
            pad_to_max_length=False,
            return_token_type_ids=True
        )
        outputs = self.tokenizer.encode_plus(
            trg_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=False,
            return_token_type_ids=True
        )
        self.index = self.index[1:]
        self.current_sample_num-=1
        if self.current_sample_num<=0:
            self.get_nraw_data()
        return {
            "input_ids":torch.tensor(inputs['input_ids'], dtype=torch.long),
            "input_mask":torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "mask_ids":torch.tensor(hard_mask["input_ids"], dtype=torch.long),
            "input_token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "target": torch.tensor(outputs["input_ids"], dtype=torch.long),
            "target_mask":torch.tensor(outputs["attention_mask"], dtype=torch.long),
            "label":torch.tensor(label, dtype=torch.long)
        }

def collate_fn(data):
    _data = data[:]
    data = dict()
    for i, d in enumerate(_data):
        for k,v in d.items():
            data_list = data.get(k, [])
            data_list.append(v)
            data[k] = data_list
    input_ids = data["input_ids"]
    input_mask = data["input_mask"]
    input_token_type_ids = data["input_token_type_ids"]
    target = data["target"]
    target_mask = data["target_mask"]
    label = data["label"]
    mask_ids = data["mask_ids"]
    
    def merge(seqs):
        lens = [len(seq) for seq in seqs]
        
        padded_seqs =  torch.zeros(len(seqs),max(lens),dtype=seqs[0].dtype)
        for i, seq in enumerate(seqs):
            end = lens[i]
#             print(padded_seqs.dtype, seq.dtype)
            if seq.dtype == torch.int64:
                padded_seqs[i,:end] = torch.LongTensor(seq[:end])
            else:
                padded_seqs[i,:end] = torch.Tensor(seq[:end])
        return padded_seqs, torch.LongTensor(lens)
    
    input_ids, input_ids_len = merge(input_ids)
    input_mask, _ = merge(input_mask)
    input_token_type_ids,_ = merge(input_token_type_ids)
    target, target_len = merge(target)
    target_mask, _ = merge(target_mask)
    label, label_len = merge(label)
    mask_ids , _ = merge(mask_ids)
    return { "input_ids":input_ids,
            "input_ids_len":input_ids_len,
            "input_mask":input_mask,
            "input_token_type_ids":input_token_type_ids,
            "target":target,
            "target_len": target_len,
            "target_mask": target_mask,
            "label": label,
            "label_len": label_len,
            "mask_ids":mask_ids
    }




if __name__ == "__main__":
    # 数据集
    DATA_DIR = "./toy_data"
    TRAIN_BATCH_SIZE = 4
    bert_base_path = "/workspace/models/transformers/pretrain/bert-base-chinese/"
    tokenizer = BertTokenizer.from_pretrained(bert_base_path) 

    train_src_file = os.path.join(DATA_DIR,"train.src")
    train_trg_file = os.path.join(DATA_DIR,"train.trg")
    train_label_file = os.path.join(DATA_DIR,"train.label")

    valid_src_file = os.path.join(DATA_DIR,"valid.src")
    valid_trg_file = os.path.join(DATA_DIR,"valid.trg")
    valid_label_file =os.path.join(DATA_DIR,"valid.label")

      

    train_dataset = CustomDataset(train_src_file, train_trg_file, train_label_file, 100000, tokenizer, MAX_LEN, True)
    valid_dataset = CustomDataset(valid_src_file, valid_trg_file, valid_label_file, 10000, tokenizer, MAX_LEN, False)
    train_params = {'collate_fn':collate_fn,
                    'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    valid_params = train_params = {'collate_fn':collate_fn,
                    'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }

    train_iter = DataLoader(train_dataset, **train_params)
    valid_iter = DataLoader(valid_dataset, **valid_params)
