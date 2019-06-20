import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Simple LSTM model which encodes both the words and the pos tags, concats them, then go through a feed forward net, before outputing the scores for each class. Softmax is not applied here, instead it's applied in loss function.
class SimpleLSTM(nn.Module):
    def __init__(self, demb, demb_pos, dmodel, dff, voc_sz, pos_voc_sz, max_len, out_class, dropout=0.5):
        super(SimpleLSTM, self).__init__()
        
        self.bidir = True
        
        self.out_class = out_class
        
        self.encoder = nn.Sequential(
            nn.Embedding(voc_sz, demb),
            nn.LSTM(demb, dmodel, num_layers=1, batch_first=True, bidirectional=self.bidir)
        )
        
        self.pos_encoder = nn.Sequential(
            nn.Embedding(pos_voc_sz, demb_pos),
            nn.LSTM(demb_pos, dmodel, num_layers=1, batch_first=True, bidirectional=self.bidir)
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.FFN = nn.Sequential(
            nn.Linear(2*dmodel*2 if self.bidir else dmodel*2,dff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dff,out_class),
        )
    
    def forward(self, x, pos):
        batch_size = x.shape[0]
        
        msg_hid, (h,c) = self.encoder(x) #torch.Size([128, 80, 240]) torch.Size([2, 128, 120])
        pos_hid, (h,c) = self.pos_encoder(pos)
        
        pooled, _ = torch.max(msg_hid, dim=1)
        pos_pooled, _ = torch.max(pos_hid, dim=1)
        
        pooled = self.dropout(pooled) # torch.Size([128, 240])
        pos_pooled = self.dropout(pos_pooled) # torch.Size([128, 240])
        
        h_sent = torch.cat((pooled, pos_pooled), dim=-1) # torch.Size([128, 480])
        #h_sent = torch.cat((pooled, pooled), dim=-1) # torch.Size([128, 480])
        
        logits = self.FFN(h_sent)
        return logits
    
    
# Enhanced model which similarly encodes the sentence and pos tag, but an additional self attention layer is introduced, so that the model can adjust its focus of words.
class SelfAttnLSTM(nn.Module):
    def __init__(self, demb, demb_pos, dmodel, dff, voc_sz, pos_voc_sz, max_len, out_class, dropout=0.5):
        super(SelfAttnLSTM, self).__init__()
        
        self.bidir = True
        
        self.out_class = out_class
        
        self.encoder = nn.Sequential(
            nn.Embedding(voc_sz, demb),
            nn.LSTM(demb, dmodel, num_layers=1, batch_first=True, bidirectional=self.bidir)
        )
        
        self.pos_encoder = nn.Sequential(
            nn.Embedding(pos_voc_sz, demb_pos),
            nn.LSTM(demb_pos, dmodel, num_layers=1, batch_first=True, bidirectional=self.bidir)
        )
        
        self.attention_proj = nn.Linear(2*dmodel*2 if self.bidir else dmodel*2, 2*dmodel*2 if self.bidir else dmodel*2)
        self.attention_softmax = nn.Softmax(dim=2) #(batch, from, to)
        self.dropout = nn.Dropout(p=dropout)
        
        self.FFN = nn.Sequential(
            nn.Linear(2*dmodel*2 if self.bidir else dmodel*2,dff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dff,out_class),
        )
    
    def forward(self, x, pos):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        msg_hid, (h,c) = self.encoder(x) #torch.Size([128, 80, 240]) torch.Size([2, 128, 120])
        pos_hid, (h,c) = self.pos_encoder(pos)
        
        feat_hid = torch.cat((msg_hid, pos_hid), dim=2)
        
        #### self attention
        feat_hid = self.attention_proj(feat_hid)
        attention = torch.bmm(feat_hid, feat_hid.transpose(1, 2)).view(batch_size, seq_len, seq_len)
        attention = self.attention_softmax(attention)
        
        #### attented vector
        att_repr = torch.bmm(attention, feat_hid) # 128,80,480
        
        pooled, _ = torch.max(att_repr, dim=1)
        
        pooled = self.dropout(pooled) # torch.Size([128, 240])
                
        logits = self.FFN(pooled)
        return logits
    
    
# This is a module for convenient use of loss, accuracy and f1 computing. Focalloss is borrowed from https://github.com/clcarwin/focal_loss_pytorch
from focalloss import FocalLoss
class Criterion:
    def __init__(self, task, cuda=True):
        self.task = 2
        self.floattensor = "torch.FloatTensor"
        if self.task==2:
            self.loss_func = FocalLoss(gamma=0.75)#nn.CrossEntropyLoss()#
        else:
            self.loss_func = nn.BCEWithLogitsLoss()
            
        if cuda:
            self.loss_func = self.loss_func.cuda()
            self.floattensor = "torch.cuda.FloatTensor"

    def loss_compute(self, logits, y):
        batch_size = y.shape[0]
        if self.task==2:
            return self.loss_func(logits, y.view(batch_size))
        else:
            logits = logits.view(batch_size)
            return self.loss_func(logits, y.view(batch_size).type(self.floattensor))
        

    def accu_compute(self, logits, y):
        batch_size = y.shape[0]
        if self.task==2:
            _, predict = torch.max(logits, dim=1)
        else:
            predict = (logits >= 0.5)
        
        y = y.type(predict.dtype)
        
        comp = (predict.view(batch_size,-1)==y.view(batch_size,-1)).type(self.floattensor)
        accu = torch.mean(comp)
        return accu.item()

    def f1_compute(self, logits, y):
        # everyone uses macro, but class imbalance should use micro
        batch_size = y.shape[0]
        if self.task==2:
            _, predict = torch.max(logits, dim=1)
        else:
            predict = (logits >= 0.5)

        y = y.type(predict.dtype)

        # to cpu
        y = y.cpu()
        predict = predict.cpu()

        f1 = f1_score(y_true=y.view(batch_size,-1), y_pred=predict.view(batch_size,-1), average='macro')

        return f1
