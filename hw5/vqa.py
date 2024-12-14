import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################
# Tokenizer
#####################################
class SimpleTokenizer:
    def __init__(self, max_len=20):
        self.max_len = max_len
        self.idx2token = {}
        self.token2idx = {}

        # Special tokens
        self.special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
        for token in self.special_tokens:
            idx = len(self.idx2token)
            self.idx2token[idx] = token
            self.token2idx[token] = idx
        
        # Example vocabulary (characters)
        vocab = list("abcdefghijklmnopqrstuvwxyz0123456789,;.!? ")
        for c in vocab:
            if c not in self.token2idx:
                idx = len(self.idx2token)
                self.idx2token[idx] = c
                self.token2idx[c] = idx

    def tokenize(self, sentence):
        # Character-level tokenization
        tokens = ['<start>']
        for c in sentence:
            if c in self.token2idx:
                tokens.append(c)
            else:
                tokens.append('<unk>')
        tokens.append('<end>')
        return [self.token2idx[t] for t in tokens]

    def pad_sequence(self, sequence):
        if len(sequence) < self.max_len:
            sequence = [self.token2idx['<pad>']] * (self.max_len - len(sequence)) + sequence
        else:
            sequence = sequence[:self.max_len]
        return sequence

    def decode(self, tokens):
        result = []
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.flatten()
            tokens = tokens.tolist()
        for t in tokens:
            if t in self.idx2token:
                tok = self.idx2token[t]
                if tok not in ['<start>', '<end>', '<pad>']:
                    result.append(tok)
                if tok == '<end>':
                    break
        return ''.join(result).strip()

#####################################
# Dataset
#####################################
class VQADataset(Dataset):
    def __init__(self, img_dir, img_prefix, question_json, answer_json, transform, tokenizer, max_a_len=5):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_prefix = img_prefix
        self.max_a_len = max_a_len

        with open(question_json, 'r') as f:
            q_data = json.load(f)
        with open(answer_json, 'r') as f:
            a_data = json.load(f)

        self.questions_data = q_data['questions']
        self.answers_data = a_data['annotations']

        self.questions = {}
        self.answers = {}
        self.img_ids = set()

        # Assuming one question and one answer per image_id for simplicity
        for question_data in self.questions_data:
            qid = question_data['image_id']
            question = question_data['question']
            self.questions[qid] = question
            self.img_ids.add(qid)

        for answer_data in self.answers_data:
            qid = answer_data['image_id']
            answer = answer_data['multiple_choice_answer']
            self.answers[qid] = answer
            self.img_ids.add(qid)

        self.img_ids = [qid for qid in self.img_ids if qid in self.questions and qid in self.answers]
        self.img_ids.sort()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        qid = self.img_ids[idx]
        img_prefix = self.img_prefix + '000'

        question = self.questions[qid].lower().strip()
        answer = self.answers[qid].lower().strip()

        if len(str(qid)) < 9:
            img_prefix += '0' * (9 - len(str(qid)))
        img_path = os.path.join(self.img_dir, img_prefix+str(qid)+'.png')

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Tokenize question
        q_tokens = self.tokenizer.tokenize(question)
        q_indices = self.tokenizer.pad_sequence(q_tokens)

        # Tokenize answer
        a_tokens = self.tokenizer.tokenize(answer)
        a_indices = self.tokenizer.pad_sequence(a_tokens)

        q_indices = torch.tensor(q_indices, dtype=torch.long)
        a_indices = torch.tensor(a_indices, dtype=torch.long)

        return img, q_indices, a_indices

#####################################
# Model
#####################################
class VQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=256, num_layers=1):
        super(VQAModel, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.fc = nn.Identity()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # We will provide: image feature + question token + previous answer token at each step
        # But for simplicity, we'll do: image + question once, then unroll answer tokens
        # Let's store image+question as a context vector (like initial hidden state)

        # A simple approach: encode question once using LSTM and combine with image
        self.q_lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.img_fc = nn.Linear(512, hidden_size)

        # Decoder LSTM for answer tokens
        self.dec_lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, img, q_indices, a_indices=None, teacher_forcing=True, max_len=20):
        batch_size = img.size(0)
        # Encode image
        img_feat = self.resnet(img)  # (batch, 512)
        img_feat = self.img_fc(img_feat).unsqueeze(1)  # (batch, 1, hidden_size)

        # Encode question
        q_embed = self.embedding(q_indices)  # (batch, q_len, emb_dim)
        _, (q_h, q_c) = self.q_lstm(q_embed)  # q_h,q_c: (1, batch, hidden_size)

        # Initialize decoder hidden state with q_h + img_feat
        dec_h = q_h + img_feat.transpose(0,1)  # (1, batch, hidden_size)
        dec_c = q_c

        if a_indices is not None and teacher_forcing:
            # Teacher forcing mode
            # a_indices: (batch, answer_seq_len)
            # Shift answer tokens for input and target
            dec_input = a_indices[:, :-1]  # (batch, seq_len-1)
            dec_target = a_indices[:, 1:]  # (batch, seq_len-1)

            dec_embed = self.embedding(dec_input)  # (batch, seq_len-1, emb_dim)
            out, _ = self.dec_lstm(dec_embed, (dec_h, dec_c))  # (batch, seq_len-1, hidden_size)
            logits = self.fc(out)  # (batch, seq_len-1, vocab_size)

            return logits, dec_target
        else:
            # Inference mode (autoregressive)
            # Start from the <start> token
            start_idx = 1  # Assuming <start> = 1
            dec_input_tok = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=img.device)
            dec_hid = (dec_h, dec_c)

            outputs = []
            for t in range(max_len):
                dec_embed_step = self.embedding(dec_input_tok)  # (batch, 1, emb_dim)
                out, dec_hid = self.dec_lstm(dec_embed_step, dec_hid)
                logits = self.fc(out)  # (batch,1,vocab_size)
                pred = torch.argmax(logits, dim=2)  # (batch,1)
                outputs.append(pred)
                dec_input_tok = pred

                # Check if all predictions are <end>
                # Assuming <end> = 2
                if (pred == 2).all():
                    break

            outputs = torch.cat(outputs, dim=1)  # (batch, seq_len)
            return outputs, None


#####################################
# Training & Evaluation
#####################################
def train_model(model, dataloader, valid_dataloader, optimizer, criterion, epochs=5, tokenizer=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, q_indices, a_indices in dataloader:
            imgs = imgs.to(device)
            q_indices = q_indices.to(device)
            a_indices = a_indices.to(device)

            optimizer.zero_grad()
            logits, dec_target = model(imgs, q_indices, a_indices, teacher_forcing=True)
            # logits: (batch, seq_len-1, vocab_size)
            # dec_target: (batch, seq_len-1)
            logits = logits.reshape(-1, logits.size(-1))
            dec_target = dec_target.reshape(-1)
            
            loss = criterion(logits, dec_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_train_loss = total_loss/len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")
        evaluate_model(model, valid_dataloader, criterion, tokenizer)

def evaluate_model(model, dataloader, criterion, tokenizer=None):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, q_indices, a_indices in dataloader:
            imgs = imgs.to(device)
            q_indices = q_indices.to(device)
            a_indices = a_indices.to(device)

            logits, dec_target = model(imgs, q_indices, a_indices, teacher_forcing=True)
            logits = logits.reshape(-1, logits.size(-1))
            dec_target = dec_target.reshape(-1)
            
            loss = criterion(logits, dec_target)
            total_loss += loss.item() * imgs.size(0)
    avg_val_loss = total_loss/len(dataloader.dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Optional: Show sample predictions
    correct = 0
    total = 0
    if tokenizer is not None:
        model.eval()
        sample_imgs, sample_q, sample_a = next(iter(dataloader))
        sample_imgs, sample_q = sample_imgs.to(device), sample_q.to(device)
        predicted_tokens, _ = model(sample_imgs, sample_q, None, teacher_forcing=False, max_len=10)
        # Decode first sample
        pred_text = tokenizer.decode(predicted_tokens[0].cpu().tolist())
        print(f"Predicted: {pred_text}, Pred Tokens: {predicted_tokens.cpu().tolist()}")
        gt_text = tokenizer.decode(sample_a[0].cpu().tolist())
        q_text = tokenizer.decode(sample_q[0].cpu().tolist())
        print(f"Sample Q: {q_text}")
        print(f"GT Answer: {gt_text}")
        print(f"Predicted: {pred_text}")
    
    for imgs, q_indices, a_indices in dataloader:
        imgs = imgs.to(device)
        q_indices = q_indices.to(device)
        a_indices = a_indices.to(device)
        predicted_tokens, _ = model(imgs, q_indices, None, teacher_forcing=False, max_len=10)
        for idx in range(len(predicted_tokens)):
            pred_text = tokenizer.decode(predicted_tokens[idx].cpu().tolist())
            gt_text = tokenizer.decode(a_indices[idx].cpu().tolist())
            if pred_text == gt_text:
                correct += 1
            total += 1
    print(f"Accuracy: {correct/total:.4f}")
    print(f"Correct {correct}, Total: {total}")
    model.train()
    return avg_val_loss