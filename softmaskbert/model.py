import transformers
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
# from transformers.modeling_bert import *
from torch.nn import functional as F

class SoftMaskModel(nn.Module):
    def __init__(self,bert_base_path,LABEL_NUM=2, VOCAB_SIZE=21128,d_detector=512, d_model=768, Dropout=0.3):
        super(SoftMaskModel, self).__init__()
        self.VOCAB_SIZE = VOCAB_SIZE
        
        self.bert_encoder = BertModel.from_pretrained(bert_base_path)
        self.embed = self.bert_encoder.embeddings
        self.lstm_encoder = nn.LSTM(d_model, 
                               d_detector,
                               num_layers=2,
                               bidirectional=True,
                              batch_first = True)
        self.detector_dropout_layer = nn.Dropout(Dropout)
        self.detector = nn.Linear(2*d_detector, LABEL_NUM )
        self.generator_dropout_layer = nn.Dropout(Dropout)
        self.generator = nn.Linear(d_model, VOCAB_SIZE)
    
    def soft_embed(self,
                   mask_embed, 
                   inputs_embed, 
                   neg_prob, 
                   pos_prob):
        neg_prob = neg_prob.unsqueeze(-1)
        pos_prob = pos_prob.unsqueeze(-1)
        embed = pos_prob*inputs_embed + neg_prob*mask_embed
        return embed
    
    def forward(self, 
                input_ids,
                input_ids_len,
                input_mask,
                input_token_type_ids,
                mask_ids
                ):
        embed_x = self.embed.word_embeddings(input_ids)
        embed_mask = self.embed.word_embeddings(mask_ids)
        # lstm encoder 
        detect_logits = self.detecting(embed_x, input_ids_len)
        detect_prob = F.softmax(detect_logits, dim=-1) # B x L * 2, 0 represents a wrong word, 1 represents a correct_word
        neg_prob = detect_prob[:,:,0]
        pos_prob = detect_prob[:,:,1]
        embed_x_with_mask = self.soft_embed(embed_mask,embed_x, neg_prob, pos_prob)
        generate_logits = self.correcting(embed_x,
                                         embed_x_with_mask,
                                         input_ids_len,
                                         input_mask,
                                         input_token_type_ids)
        
        return detect_logits, generate_logits
     
    def correcting(self, 
                   embed_x, 
                   embed_x_with_mask, 
                   input_ids_len, 
                   input_mask, 
                   input_token_type_ids):
        outputs= self.bert_encoder(attention_mask=input_mask,
                          token_type_ids = input_token_type_ids,
                          inputs_embeds=embed_x_with_mask)  #B, L, d_model 
        outputs = outputs.last_hidden_state
        # print(type(_))
        outputs = outputs+embed_x #residual
        outputs = self.generator_dropout_layer(outputs)
        outputs = self.generator(outputs)
        return outputs
    
    def detecting(self, embed_x, input_ids_len):
        self.lstm_encoder.flatten_parameters()
        total_length = embed_x.size(1)  # get the max sequence length
        lstm_embed_x = torch.nn.utils.rnn.pack_padded_sequence(embed_x,input_ids_len,batch_first=True, enforce_sorted=False)
        outputs, (hidden, _) = self.lstm_encoder(lstm_embed_x) #1， batch, hidden_size
        outputs = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True, total_length=total_length)[0]
        outputs = self.detector_dropout_layer(outputs)
        outputs = self.detector(outputs) # logits of each word is a wrong word
        return outputs