import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, precision_score

def train(model, optimizer, criterion, training_loader, device): 
    model.train()
    total_loss = 0
    t_labels = []
    t_outputs = []
    for data in tqdm(training_loader, total=len(training_loader)):
        images = data['images'].to(device)
        labels = data['labels'].to(device)
        # print(images.shape, labels.shape)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        t_labels.extend(labels.cpu().detach().numpy().tolist())
        t_outputs.extend(np.argmax(torch.sigmoid(outputs).cpu().detach().numpy(),1).tolist())
    train_loss = total_loss / len(training_loader)
    train_f1 = f1_score(t_labels, t_outputs)
    train_acc = accuracy_score(t_labels, t_outputs)
    return train_f1, train_acc, train_loss


def validation(model, criterion, validation_loader, device):
    model.eval()

    total_loss = 0
    v_labels = []
    v_outputs = []
    with torch.no_grad():
        for data in tqdm(validation_loader, total=len(validation_loader)):
            images = data['images'].to(device)
            labels = data['labels'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            v_labels.extend(labels.cpu().detach().numpy().tolist())
            v_outputs.extend(np.argmax(torch.sigmoid(outputs).cpu().detach().numpy(),1).tolist())
    val_loss = total_loss / len(validation_loader)
    val_f1 = f1_score(v_labels, v_outputs) 
    val_acc = accuracy_score(v_labels, v_outputs)

    return val_f1, val_acc, val_loss

def fit(model, epochs, optimizer, criterion, training_loader, validation_loader, save_path, device='cuda'):
    best_val_loss = 10000
    early_stopping = 4
    for epoch in range(epochs):
        train_f1, train_acc, train_loss = train(model, optimizer, criterion, training_loader, device)
        val_f1, val_acc, val_loss = validation(model, criterion, validation_loader, device)
        
        print('Epoch: {}/{} | Train Loss: {:.6f}, Train F1: {:.4f}, Train Acc: {:.4f} | Val Loss: {:.6f}, Val F1: {:.4f}, Val Acc: {:.4f}'.format(epoch+1,epochs,train_loss,train_f1,train_acc,val_loss,val_f1,val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping = 4
            # torch.save(model.state_dict(), 'saved/checkpoint.pt')
            torch.save(model, save_path)
            print("\n** Checkpoint Saved!\n")
        else:
            early_stopping -= 1
            if early_stopping <= 0:
                break


def predict(model, test_loader, device='cuda'):
    model.eval()

    test_labels = []
    test_outputs = []
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            images = data['images'].to(device)
            labels = data['labels'].to(device)

            outputs = model(images)

            test_labels.extend(labels.cpu().detach().numpy().tolist())
            test_outputs.extend(np.argmax(torch.sigmoid(outputs).cpu().detach().numpy(),1).tolist())
    
    test_f1 = f1_score(test_labels, test_outputs)
    test_acc = accuracy_score(test_labels, test_outputs)
    test_recall = recall_score(test_labels, test_outputs)
    test_precision = precision_score(test_labels, test_outputs)
    print("testing result: ")
    print("f1: ", test_f1)
    print("precision: ", test_precision)
    print("recall: ", test_recall)
    print(confusion_matrix(test_labels, test_outputs))
    return test_outputs, test_labels