import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import kobert

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import pandas as pd

#train & test 데이터로 나누기
from sklearn.model_selection import train_test_split

#model 저장
import os

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))  

class BERTClassifier(nn.Module):
    def __init__(self,
                    bert,
                    hidden_size=768,
                    num_classes=4,  ##클래스 수 조정##
                    dr_rate=None,
                    params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                                attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class FeelingClassifier():

    # GPU 사용시 아래 주석 해제
    device = torch.device("cuda:0")

    # BERT 모델, Vocabulary 불러오기
    bertmodel, vocab = get_pytorch_kobert_model()
    chatbot_data = pd.read_excel('/data/kocrawl/word-feeling-tag.xlsx')

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    def __init__(self, isTrain):

        # Setting parameters
        self.max_len = 64
        self.batch_size = 10
        self.warmup_ratio = 0.1
        self.num_epochs = 10
        self.max_grad_norm = 1
        self.log_interval = 200
        self.learning_rate = 5e-5

        self.model_loaded = False

        # /saved/CLASS_NAME/
        self.model_dir = '/data/EggChat/kochat/demo/saved/FeelingClassifier'
       
        # /saved/CLASS_NAME/CLASS_NAME.xxx
        self.model_file = '/data/EggChat/kochat/demo/saved/FeelingClassifier/FeelingClassifier'

        if isTrain:
            self.organize_data_list()
            self.train_test_split()
            self.tokenizing_dataset()
            self.use_dataloader()
            self.learn_Classifier()
            self._save_model()
            


    def organize_data_list(self):
        FeelingClassifier.chatbot_data.loc[(FeelingClassifier.chatbot_data['feeling'] == 'good'), 'feeling'] = 0  # good -> 0
        FeelingClassifier.chatbot_data.loc[(FeelingClassifier.chatbot_data['feeling'] == 'sad'), 'feeling'] = 1  # sad -> 1
        FeelingClassifier.chatbot_data.loc[(FeelingClassifier.chatbot_data['feeling'] == 'tired'), 'feeling'] = 2  # tired -> 2
        FeelingClassifier.chatbot_data.loc[(FeelingClassifier.chatbot_data['feeling'] == 'stress'), 'feeling'] = 3  # stress -> 3

        self.data_list = []
        for q, label in zip(FeelingClassifier.chatbot_data['word'], FeelingClassifier.chatbot_data['feeling']):
            data = []
            data.append(q)
            data.append(str(label))

            self.data_list.append(data)


    def train_test_split(self):
        self.dataset_train, self.dataset_test = train_test_split(self.data_list, test_size=0.25, random_state=0)
        print(len(self.dataset_train))
        print(len(self.dataset_test))



    def tokenizing_dataset(self):
        # 토큰화
        self.tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(self.tokenizer, FeelingClassifier.vocab, lower=False)

        self.data_train = BERTDataset(self.dataset_train, 0, 1, tok, self.max_len, True, False)
        self.data_test = BERTDataset(self.dataset_test, 0, 1, tok, self.max_len, True, False)

        print(self.data_train[0])

    def use_dataloader(self):
        self.train_dataloader = torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size, num_workers=5)
        self.test_dataloader = torch.utils.data.DataLoader(self.data_test, batch_size=self.batch_size, num_workers=5)

        # BERT 모델 불러오기
        FeelingClassifier.model = BERTClassifier(FeelingClassifier.bertmodel, dr_rate=0.5).to(FeelingClassifier.device)

        # optimizer와 schedule 설정
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in FeelingClassifier.model.named_parameters() if not any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in FeelingClassifier.model.named_parameters() if any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(self.optimizer_grouped_parameters, lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        t_total = len(self.train_dataloader) * self.num_epochs
        self.warmup_step = int(t_total * self.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_step,
                                                    num_training_steps=t_total)

        self.train_dataloader

    # 정확도 측정을 위한 함수 정의
    def calc_accuracy(self, X, Y):
        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
        return train_acc


    def learn_Classifier(self):
        for e in range(self.num_epochs):
            train_acc = 0.0
            test_acc = 0.0
            FeelingClassifier.model.train()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(self.train_dataloader)):
                self.optimizer.zero_grad()
                token_ids = token_ids.long().to(FeelingClassifier.device)
                segment_ids = segment_ids.long().to(FeelingClassifier.device)
                valid_length = valid_length
                label = label.long().to(FeelingClassifier.device)
                out = FeelingClassifier.model(token_ids, valid_length, segment_ids)
                loss = self.loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(FeelingClassifier.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                train_acc += self.calc_accuracy(out, label)
                if batch_id % self.log_interval == 0:
                    print(
                        "epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                           train_acc / (batch_id + 1)))
            print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

            FeelingClassifier.model.eval()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(self.test_dataloader)):
                token_ids = token_ids.long().to(FeelingClassifier.device)
                segment_ids = segment_ids.long().to(FeelingClassifier.device)
                valid_length = valid_length
                label = label.long().to(FeelingClassifier.device)
                out = FeelingClassifier.model(token_ids, valid_length, segment_ids)
                test_acc += self.calc_accuracy(out, label)
            print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))
    
    def _save_model(self):
        """
        모델을 저장장치에 저장합니다.
        """

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.model.state_dict(), self.model_file + '.pth')

# classifier = FeelingClassifier()
"""
sentence = string형 input (entity 넣어주기)
predict(sentence)
"""