#From HW1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torch.optim as optim

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

def train_model(model, num_epochs, train_loader, device='cuda', enable_defense=True, attack='pgd', eps=0.1):
    # TODO: implement this function that trains a given model on the MNIST dataset.
    # this is a general-purpose function for both standard training and adversarial training.
    # (toggle enable_defense parameter to switch between training schemes)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
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
    return success_adversaries