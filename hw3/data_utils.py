# !pip install tensorboardX
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from collections import defaultdict
from datasets import load_dataset

# from tensorboardX import SummaryWriter

def collate_fn(batch, pad_idx=0):
    #need to pad the sequences in the batch to make them the same length
    texts, labels = zip(*batch)
    max_len = max(len(t) for t in texts)
    padded = torch.stack([F.pad(t, (0, max_len - len(t)), value=pad_idx) for t in texts])
    labels = torch.stack(labels)
    return padded, labels

def load_dataloader(directory='mnist_data/', batch_size=64, only_test=False, imdb=False, word_subs=None):
    ## Dataloaders
    if not imdb:
        test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor()]
        ))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if only_test:
            return None, test_loader
        
        train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor()]
        ))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if imdb:
        test_dataset = TextDataset(split='test', word_subs=word_subs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_idx=train_dataset.pad_idx))
        if only_test:
            return None, test_loader
        
        train_dataset = TextDataset(split='train', word_subs=word_subs)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_idx=train_dataset.pad_idx))
        
        
    return train_loader, test_loader

class TextDataset(Dataset):
    def __init__(self, split='train', max_length=256, word_subs=None):
        self.dataset = load_dataset("stanfordnlp/imdb", split=split)
        self.max_length = max_length
        self.split = split
        
        self.word2idx = {}
        self.embeddings = []
        with open(r'C:\Users\ved67\UIUC\CS521\CS521_HWs\hw3\glove.6B.300d.txt', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.word2idx[word] = i
                self.embeddings.append(vector)
        
        self.pad_idx = len(self.embeddings)
        self.word2idx['<pad>'] = self.pad_idx
        self.unk_idx = len(self.embeddings) + 1
        self.word2idx['<unk>'] = self.unk_idx
        self.embeddings.append(np.zeros(300))
        self.embeddings.append(np.random.normal(size=300))
        
        self.embeddings = np.stack(self.embeddings)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        # Precompute word subs or load from file
        if word_subs is None:
            self.substitutions = self.get_word_substitutions()
        else:
            with open(word_subs, 'r') as f:
                self.substitutions = json.load(f)
        fin_subs = {}
        for k, v in self.substitutions.items():
            fin_subs[int(k)] = [int(i) for i in v]
        self.substitutions = fin_subs
    def get_word_substitutions(self):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.lm = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float16)
        self.lm = self.lm.to('cuda')
        # Find similar words using GloVe
        word_subs = defaultdict(list)
        vectors = torch.FloatTensor(self.embeddings)
        normalized = F.normalize(vectors, dim=1)
        
        topk = 8
        batched = 1024
        for batch_start in range(0, len(self.idx2word), batched):
            if batch_start % 10240 == 0:
                print(f"Batch {batch_start}/{len(self.idx2word)}")
            batch_end = min(batch_start + batched, len(self.idx2word))
            current_batch = normalized[batch_start:batch_end]
            batch_similarity = torch.mm(current_batch, normalized.t())
            sims, indices = batch_similarity.topk(topk + 1)
            for batch_idx, word_idx in enumerate(range(batch_start, batch_end)):
                word = self.idx2word[word_idx]
                
                if word in ['<pad>', '<unk>']:
                    continue
                # Filter subs using lm probs
                context = f"The movie was {word}"
                inputs = self.tokenizer(context, return_tensors="pt")
                inputs = {k: v.to(self.lm.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.lm(**inputs)
                    base_score = outputs.logits[0, -1].logsumexp(-1)
                    
                for idx in  indices[batch_idx][1:]:
                    candidate = self.idx2word[idx.item()]
                    new_context = f"The movie was {candidate}"
                    inputs = self.tokenizer(new_context, return_tensors="pt")
                    inputs = {k: v.to(self.lm.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.lm(**inputs)
                        candidate_score = outputs.logits[0, -1].logsumexp(-1)
                        
                    #print(f"{word} -> {candidate}: {candidate_score.item() - base_score.item()}")
                    if (base_score - candidate_score) < 5:  # Threshold from paper
                        word_subs[word_idx].append(idx.item())
        
        del self.tokenizer, self.lm, inputs, outputs, candidate_score, base_score
        
        with open(f'word_subs_{self.split}.json', 'w') as f:
            json.dump(word_subs, f)
        
        return word_subs

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        text, label = data['text'], data['label']
        words = text.lower().split()[:self.max_length]
        indices = [self.word2idx.get(word, 0) for word in words]
        indices = torch.LongTensor(indices)
        return indices, torch.tensor(int(label))

