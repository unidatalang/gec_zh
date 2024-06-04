 
import transformers
import torch
import torch.nn as nn
from transformers import BertTokenizer

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import functional as F
from torch import cuda
import os
import tqdm
import sys 
root_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_path)
from softmaskbert.model import SoftMaskModel 
from softmaskbert.dataset import ServeDataSet, serve_collate_fn

models = os.path.join(root_path, "models","softmaskbert")



bert_base_path = os.path.join(models,"bert-base-chinese")
model_checkpoint = os.path.join(models,"checkpoint_best.pt")

tokenizer = BertTokenizer.from_pretrained(bert_base_path)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'


def load_model_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    START_EPOCH = checkpoint['epoch']
    eval_loss = checkpoint['eval_loss']
    best_loss = checkpoint["best_loss"]
    START_STEP = checkpoint["step"]
    print("Loaded exits model ...")
    print("epoch {}, eval_loss {}, best_loss {}, step {}".format(START_EPOCH, 
                                                                eval_loss, 
                                                                best_loss, 
                                                                START_STEP))
    return START_EPOCH, best_loss, START_STEP

def load_model_from_checkpoint_to_single_gpu(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    params = dict()
    for k,v in checkpoint['model_state_dict'].items():
        params[k.replace("module.","")]=v
    model.load_state_dict(params)
    START_EPOCH = checkpoint['epoch']
    eval_loss = checkpoint['eval_loss']
    best_loss = checkpoint["best_loss"]
    START_STEP = checkpoint["step"]
    print("Loaded exits model ...")
    print("epoch {}, eval_loss {}, best_loss {}, step {}".format(START_EPOCH, 
                                                                eval_loss, 
                                                                best_loss, 
                                                                START_STEP))
    return START_EPOCH, best_loss, START_STEP


def get_model(checkpoint):
    model = SoftMaskModel(bert_base_path) 
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))>1 and device!="cpu":
        model = nn.DataParallel(model)
        load_model_from_checkpoint(model, model_checkpoint)
    else:
        load_model_from_checkpoint_to_single_gpu(model, model_checkpoint)
    model.to(device)
    model.eval()
    return model

model = get_model(model_checkpoint)

#获取一组数据
def get_data_iter(text):
    test_params =  {'collate_fn':serve_collate_fn,
                    'batch_size': 4,
                    'shuffle': False,
                    'num_workers': 0
                    }
    test_dataset = ServeDataSet(text, tokenizer)
    test_iter = DataLoader(test_dataset, **test_params)
    return test_iter

def get_pred_result(detect_logtis, cor_logits, input_ids_len, inputs_text = None, replace_unk=False, unk_symbol="[UNK]"):
    
    detect_pred = torch.argmax(detect_logtis, dim=-1)
    cor_pred = torch.argmax(cor_logits, dim=-1)
    detect_pred = detect_pred.detach().cpu().numpy()
    cor_pred = cor_pred.detach().cpu().numpy()
    result = []
    for i, l in enumerate(input_ids_len):
        label = detect_pred[i][1:l-1]
        text = tokenizer.convert_ids_to_tokens(cor_pred[i][1:l-1])
        if replace_unk and inputs_text: # replace UNK
            orig_text = inputs_text[i]
            assert len(orig_text) == len(text)
            for char_id in range(len(orig_text)):
                if text[char_id] == unk_symbol:
                    text[char_id] = orig_text[char_id]
        result.append((text, label))
    return result 

def predict_single_text(text):
    result_list = []
    with torch.no_grad():
        inputs = [list(text)]
        data_iter = get_data_iter(inputs)
        for d in data_iter:
            input_ids = d["input_ids"].to(device, dtype = torch.long)
            input_ids_len = d["input_ids_len"].numpy()
            input_mask = d["input_mask"].to(device, dtype = torch.long)
            input_token_type_ids = d["input_token_type_ids"].to(device, dtype = torch.long)
            mask_ids = d["mask_ids"].to(device, dtype = torch.long)
            detect, correct = model(input_ids,
                                input_ids_len,
                                input_mask, 
                                input_token_type_ids, 
                                mask_ids)
            result = get_pred_result(detect, correct, input_ids_len, inputs, True)
            result_list.extend(result)
    return result_list

def predict_mul_text(text):
    result_list = []
    with torch.no_grad():
        inputs = [list(text)]
        data_iter = get_data_iter(inputs)
        for d in data_iter:
            input_ids = d["input_ids"].to(device, dtype = torch.long)
            input_ids_len = d["input_ids_len"].numpy()
            input_mask = d["input_mask"].to(device, dtype = torch.long)
            input_token_type_ids = d["input_token_type_ids"].to(device, dtype = torch.long)
            mask_ids = d["mask_ids"].to(device, dtype = torch.long)
            detect, correct = model(input_ids,
                                input_ids_len,
                                input_mask, 
                                input_token_type_ids, 
                                mask_ids)
            result = get_pred_result(detect, correct, input_ids_len, inputs, True)
            result_list.extend(result)
    return result_list

def correct_text(text):
    cor_res = predict_single_text(text)
    cor_text = "".join(cor_res[0][0])
    assert len(cor_text) == len(text)
    ret = dict(original_text=text,
               corrected_text = cor_text)
    details = []
    for ind in range(len(text)):
        ori_char = text[ind]
        cor_char = cor_text[ind]
        if ori_char != cor_char:
            details.append({"pos":ind, "ori":ori_char, "cor":cor_char})
    ret["details"] = details
    return ret


if __name__ == "__main__":
    text = "你知道金子塔，在那里吗，是在奥晕会上吗？"
    print(correct_text(text))
    
            





