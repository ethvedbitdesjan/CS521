import os
from data_utils import *
from model_utils import *
from interval_model import *
from train_test import *
import gc
import numpy as np

word_subs = 'hw3/word_subs1.json'

train_dataloader, test_dataloader = load_dataloader('', imdb=True, word_subs=word_subs, batch_size=32)
base_model = BOWModel(train_dataloader.dataset.embeddings, train_dataloader.dataset.pad_idx)
base_model.to('cuda')
# state_dict = torch.load('base_model1.pth')
# base_model.load_state_dict(state_dict)
interval_model = IntervalModel(base_model, mur_computation=True).to('cuda')

train_text_verified(base_model, interval_model, train_dataloader, train_dataloader.dataset, num_epochs=100, elide_last=True)