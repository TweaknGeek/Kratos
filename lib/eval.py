import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
import os


def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.to('cpu').tolist()
            y_pred += predicted.to('cpu').tolist()
    accuracy = sum(int(a == p) for a, p in zip(y_true, y_pred)) / len(y_true)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return total_loss / len(test_loader.dataset), accuracy, precision, recall, f1


def lr_schedule(epoch, optimizer):
    if epoch < 10:
        lr = 0.001
    elif epoch < 20:
        lr = 0.0005
    elif epoch < 30:
        lr = 0.0001
    else:
        lr = 0.00005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, epoch, optimizer, loss):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, 'models/model_{}.pth'.format(epoch))


def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def early_stopping(model, optimizer, early_stop_count, val_loss, best_val_loss):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, epoch, optimizer, val_loss)
        early_stop_count = 0
    else:
        early_stop_count += 1
    if early_stop_count >= 5:
        print('Early stopping...')
        return True
    return False


def train_model_with_eval(model, train_loader, val_loader, num_epochs, learning_rate=0.001):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    early_stop_count = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        lr_schedule(epoch, optimizer)
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        val_loss, accuracy, precision, recall, f1 = evaluate(model, val_loader)
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss / len(train_loader.dataset):.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        if early_stopping(model, optimizer, early_stop_count, val_loss, best_val_loss):
            break
    return model


if __name__ == '__main__':
    model = CelebAModel()
    model = model.to('cuda')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    train_model_with_eval(model, train_loader, val_loader, num_epochs)
