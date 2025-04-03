from classifier import *
#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import kobert

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

import os

class Predictor(FeelingClassifier):
    def __init__(self):
        super().__init__(False)
        self.model_loaded = False
        self.model = FeelingClassifier(False).model
        
        # /saved/CLASS_NAME/
        self.model_dir = '/data/EggChat/kochat/demo/saved/FeelingClassifier'

        # /saved/CLASS_NAME/CLASS_NAME.xxx
        self.model_file = '/data/EggChat/kochat/demo/saved/FeelingClassifier/FeelingClassifier'


    def _load_model(self):
        """
        저장된 모델을 불러옵니다.
        """

        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")

        if not self.model_loaded:
            self.device = torch.device("cuda:0")
                        #CPU에서 불러올건지, GPU에서 불러올건지 고민 후 수정
            self.model.load_state_dict(torch.load(self.model_file + '.pth'))
            self.model_loaded = True
        
        
    def predict(self, predict_sentence)->str:

        #BERT 모델, Vocabulary 불러오기
        bertmodel, vocab = get_pytorch_kobert_model()
        #토큰화
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        data = [predict_sentence, '0']
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, tok, self.max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=self.batch_size, num_workers=5)
        
        self._load_model()

        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)

            valid_length = valid_length
            label = label.long().to(self.device)

            out = self.model(token_ids, valid_length, segment_ids)
  
            test_eval = []

            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()

                if np.argmax(logits) == 0:
                    test_eval.append("good")
                elif np.argmax(logits) == 1:
                    test_eval.append("sad")
                elif np.argmax(logits) == 2:
                    test_eval.append("tired")
                elif np.argmax(logits) == 3:
                    test_eval.append("stress")

            print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

        return test_eval[0]