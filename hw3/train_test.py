#From HW1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torch.optim as optim
from interval_model import IntervalModel, Interval
import time

# The last argument 'targeted' can be used to toggle between a targeted and untargeted attack.
def fgsm(model, x, y, eps):
    #TODO: implement this as an intermediate step of PGD
    # Notes: put the model in eval() mode for this function
    
    x = x.detach().requires_grad_(True) #need the gradient
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    grad = x.grad.data
    x = x + eps*grad.sign() #untargeted, move to maximize loss
    return x

def pgd_untargeted(model, x, y, k, eps, eps_step):
    #TODO: implement this 
    # Notes: put the model in eval() mode for this function
    model.eval()

    min_x = x - eps
    max_x = x + eps
    for i in range(k):
        x = fgsm(model, x, y, eps_step)
        x = torch.clamp(x, min=min_x, max=max_x)
        # preds = model(x).argmax()
        # if preds != y:
        #     break early?
    return x

def train_model(model, num_epochs, train_loader, device='cuda', enable_defense=True, attack='pgd', eps=0.1, lr=0.001, weight_decay=0.01):
    # TODO: implement this function that trains a given model on the MNIST dataset.
    # this is a general-purpose function for both standard training and adversarial training.
    # (toggle enable_defense parameter to switch between training schemes)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    eps_step = 0.01
    k = 25
    for num_epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ce_loss = loss(logits, y)
            epoch_loss += ce_loss.item()
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            if enable_defense:
                #adversarial training
                if attack == 'fgsm':
                    x = fgsm(model, x, y, eps) #could also use eps_step here
                elif attack == 'pgd':
                    x = pgd_untargeted(model, x, y, k, eps, eps_step)
                else:
                    raise ValueError(f"Attack {attack} not implemented")

                logits = model(x)
                ce_loss = loss(logits, y)
                epoch_loss += ce_loss.item()
                optimizer.zero_grad()
                ce_loss.backward()
                optimizer.step()
        print(f"Epoch {num_epoch} done, Epoch Loss {epoch_loss}, Time Taken {time.time()-start_time}")

def test_model(model, x, y):
    #assumes model is in eval mode, same device x, y, model

    preds = model(x).argmax(dim=1)
    correct += (preds == y).sum().item()
    total += y.size(0)
    return correct, total

def test_model_on_attacks(model, test_loader, device='cuda', attack='pgd', eps=0.1):
    # TODO: implement this function to test the robust accuracy of the given model
    # use pgd_untargeted() within this function
    model.eval()
    vanilla_correct = 0
    attack_correct = 0
    vanilla_total = 0
    attack_total = 0
    robust_correct = 0
    robust_total = 0
    success_adversaries = []

    k = 25
    eps_step = eps
    for i, batch in enumerate(test_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        if attack == 'fgsm':
            attack_x = fgsm(model, x, y, eps)
        elif attack == 'pgd':
            eps_step = eps/k
            attack_x = pgd_untargeted(model, x, y, k, eps, eps_step)
        else:
            raise ValueError(f"Attack {attack} not implemented")

        preds = model(x).argmax(dim=1)
        attack_preds = model(attack_x).argmax(dim=1)

        vanilla_correct += (preds == y).sum().item()
        attack_correct += (attack_preds == y).sum().item()
        robust_correct += ((attack_preds == y)&(preds==y)).sum().item()
        vanilla_total += y.size(0)
        attack_total += y.size(0)
        robust_total += y.size(0)

        adversaries = ((attack_preds != y)&(preds==y)).nonzero().squeeze()
        if len(adversaries.shape) > 0:
            success_adversaries.extend([(attack_x[i].detach().cpu(), attack_preds[i].detach().cpu(), x[i].detach().cpu(), y[i].detach().cpu()) for i in adversaries])
    print(f"Testing Vanilla Accuracy: {vanilla_correct/vanilla_total}")
    print(f"Testing Attack Accuracy: {attack_correct/attack_total}")
    print(f"Testing Robust Accuracy: {robust_correct/robust_total}")
    return success_adversaries, vanilla_correct/vanilla_total, attack_correct/attack_total, robust_correct/robust_total

def get_verified_loss(interval_model, data_interval, target, criterion, elide_last=False, return_worst_case=True, skip_layers=0):
    if elide_last:
        batch_size = target.size(0)
        device = 'cuda'
        penultimate_bounds = interval_model(data_interval, absorb_last=True, skip_layers=skip_layers)
        last_layer = interval_model.model[-1]
        num_classes = last_layer.W.size(1)
        W = last_layer.W.t()  # [out_features, in_features]
        b = last_layer.b  # [out_features]
        W_batch = W.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, out_features, in_features]
        b_batch = b.unsqueeze(0).expand(batch_size, -1)  # [batch, out_features]
        W_correct = W_batch[torch.arange(batch_size), target].unsqueeze(2)  # [batch, in_features, 1, 1]
        b_correct = b_batch[torch.arange(batch_size), target].unsqueeze(1)  # [batch, 1]
        mask = torch.ones((batch_size, num_classes), device=target.device).bool()
        mask[torch.arange(batch_size), target] = False
        #print(W_batch.shape, mask.shape)
        W_wrong = W_batch[mask].view(batch_size, num_classes-1, -1).transpose(1, 2)  # [batch, in_features, num_classes-1]
        W_spec = W_wrong - W_correct  # [batch, in_features, num_classes-1]
        b_wrong = b_batch[mask].view(batch_size, num_classes-1)  # [batch, num_classes-1]
        b_spec = b_wrong - b_correct  # [batch, num_classes-1]

        center = (penultimate_bounds.lower + penultimate_bounds.upper) / 2  # [batch, in_features]
        radius = (penultimate_bounds.upper - penultimate_bounds.lower) / 2  # [batch, in_features]
        
        #[batch, in_features] @ [batch, in_features, num_classes-1]
        mu = torch.bmm(center.unsqueeze(1), W_spec).squeeze(1) + b_spec # [batch, num_classes-1]
        r = torch.bmm(radius.unsqueeze(1), torch.abs(W_spec)).squeeze(1) # [batch, num_classes-1]
        worst_case = mu + r
        worst_case_output = torch.zeros((batch_size, num_classes), device=target.device)
        worst_case_output[mask] = worst_case.view(-1) # [batch, num_classes]
        #worst_case_output[~mask] = mu[~mask] - r[~mask]
        verified_loss = criterion(worst_case_output, target)
    else:
        interval_output = interval_model.forward(data_interval, skip_layers=skip_layers)
        worst_case_output = torch.zeros_like(interval_output.lower) #N, C
        worst_case_output.scatter_(1, target.unsqueeze(1), interval_output.lower.gather(1, target.unsqueeze(1))) #worst_case_output[0, t0] = interval_output.lower[0, t0]
        mask = torch.ones_like(interval_output.upper).scatter_(1, target.unsqueeze(1), 0) #mask[0, t0] = 0
        worst_case_output += mask * interval_output.upper
        verified_loss = criterion(worst_case_output, target)
    
    if not return_worst_case:
        return verified_loss
    
    return verified_loss, worst_case_output

def train_verified(model: nn.Module, interval_model: IntervalModel, num_epochs, train_loader, device='cuda', eps=0.1, lr=0.001, weight_decay=0.01, elide_last=False):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    epoch_times = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        
        # Linear schedules
        kappa = max(1.0 - 0.5 * epoch / (num_epochs -1), 0.5)
        current_eps = min(eps * epoch / (num_epochs -1), eps)
            
        total_loss = 0
        correct = 0
        verified_correct = 0
        total = 0
        
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            clean_output = model(data)
            
            if current_eps > 0:
                # data_interval = Interval(
                #     lower=torch.clamp(data - current_eps, min=0, max=1),
                #     upper=torch.clamp(data + current_eps, min=0, max=1)
                # )
                data_interval = Interval(lower=data - current_eps, upper=data + current_eps)
                verified_loss, worst_case_output = get_verified_loss(interval_model, data_interval, target, criterion, elide_last=elide_last)
            else:
                verified_loss = 0
            natural_loss = criterion(clean_output, target)
            loss = kappa * natural_loss + (1 - kappa) * verified_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            pred = clean_output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
            if current_eps > 0:
                verified_correct += worst_case_output.argmax(dim=1).eq(target).sum().item()
            
            total += batch_size
            
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        verified_accuracy = 100. * verified_correct / total if current_eps > 0 else 0
        epoch_times.append(time.time() - start_time)
        print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%, '
              f'Verified Accuracy: {verified_accuracy:.2f}%, '
              f'epsilon: {current_eps:.4f}, kappa: {kappa:.2f} '
              f'time for epoch: {round(epoch_times[-1], 2)}')
    return epoch_times 
    
def train_text_verified(model, interval_model, data_loader, dataset, num_epochs=100, device='cuda', eps=0.1, lr=0.001, weight_decay=0.01, elide_last=False, test_mode=False):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    epoch_times = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        kappa = max(1.0 - 0.5 * epoch / (num_epochs - 1), 0.5) if not test_mode else 1.0
        current_eps = min(eps * epoch / (num_epochs - 1), eps) if not test_mode else eps
        
        total_loss = 0
        correct = 0
        verified_correct = 0
        total = 0
        
        for i, (texts, labels) in enumerate(data_loader):
            texts, labels = texts.to(device), labels.to(device)
            batch_size = texts.size(0)
            clean_output = model(texts)
            pad_mask = (texts != model.pad_idx).bool().cuda()
            
            if current_eps > 0: #For text, eps is not used as with continuous data, so we use this as a flag for verification training enabled
                embeddings = model.get_transformed_embedding(texts)  # B, L, D
                lower_bounds = []
                upper_bounds = []
                for b in range(batch_size):
                    word_subs = []
                    for l in range(texts.size(1)):
                        word_idx = texts[b, l].item()
                        if word_idx in dataset.substitutions:
                            subs = list(dataset.substitutions[word_idx]+[word_idx])
                            sub_indices = torch.LongTensor(subs).to(device)
                            word_subs.append(sub_indices)
                            # transformed_subs = model.get_transformed_embedding(sub_indices)
                            # lower_bound = torch.min(transformed_subs, dim=0)[0]
                            # upper_bound = torch.max(transformed_subs, dim=0)[0]
                        else:
                            word_subs.append(torch.tensor([word_idx]).to(device))
                            # transformed = model.get_transformed_embedding(
                            #     torch.tensor([word_idx]).to(device))[0]
                            # lower_bound = upper_bound = transformed
                    max_len = max([len(sub) for sub in word_subs])
                    pad_subs = torch.nn.utils.rnn.pad_sequence(word_subs, batch_first=True, padding_value=model.pad_idx)
                    transformed_subs = model.get_transformed_embedding(pad_subs)
                    pad_subs_mask = (pad_subs != model.pad_idx)

                    dummy_value = torch.max(transformed_subs) + 1  # A value larger than any real value
                    masked_tensor = torch.where(pad_subs_mask.unsqueeze(-1), transformed_subs, dummy_value)
                    lower_bound = torch.min(masked_tensor, dim=1)[0]

                    dummy_value = torch.min(transformed_subs) - 1  # A value smaller than any real value
                    masked_tensor = torch.where(pad_subs_mask.unsqueeze(-1), transformed_subs, dummy_value)
                    upper_bound = torch.max(masked_tensor, dim=1)[0]
                    
                    center = embeddings[b]
                    assert torch.all(lower_bound[pad_mask[b]] - center[pad_mask[b]] < 1e-5) and torch.all(center[pad_mask[b]] - upper_bound[pad_mask[b]] < 1e-5)
                    lower_bound = center - current_eps * (center - lower_bound) #use epsilon as a scaling factor
                    upper_bound = center + current_eps * (upper_bound - center)
                    lower_bounds.append(lower_bound)
                    upper_bounds.append(upper_bound)
                upper_bounds = torch.stack(upper_bounds)
                lower_bounds = torch.stack(lower_bounds)
                pad_mask = (texts != model.pad_idx).bool().cuda()
                upper_bounds = upper_bounds * pad_mask.unsqueeze(-1).float()
                lower_bounds = lower_bounds * pad_mask.unsqueeze(-1).float()
                sum_up = torch.sum(upper_bounds, dim=1)
                sum_low = torch.sum(lower_bounds, dim=1)
                denominator = torch.sum(pad_mask, dim=1, keepdim=True).clamp(min=1e-8)
                mean_up = sum_up / denominator #B, D
                mean_low = sum_low / denominator #B, D
                
                embed_interval = Interval(mean_low, mean_up)
                verified_loss, worst_case_output = get_verified_loss(interval_model, embed_interval, labels, criterion, elide_last=elide_last, skip_layers=3)
            else:
                verified_loss = 0
                
            natural_loss = criterion(clean_output, labels)
            loss = kappa * natural_loss + (1 - kappa) * verified_loss
            
            if not test_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * batch_size
            pred = clean_output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            
            if current_eps > 0:
                verified_correct += worst_case_output.argmax(dim=1).eq(labels).sum().item()
            
            total += batch_size
        print(f"Time for loop: {time.time()-start_time}")
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        verified_accuracy = 100. * verified_correct / total if current_eps > 0 else 0
        epoch_times.append(time.time() - start_time)
        print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%, '
              f'Verified Accuracy: {verified_accuracy:.2f}%, '
              f'epsilon: {current_eps:.4f}, kappa: {kappa:.2f}, '
              f'Time: {time.time() - start_time:.2f}s')
    
    return epoch_times