import torch
from torchtext import data
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

def train_model(model, criterion, optimizer, scheduler, dataiter_dict, dataset_sizes, batch_size, num_epochs=25):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 200

    val_loss = []
    train_loss = []
    val_acc = []
    train_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            sentiment_corrects = 0
            tp = 0.0 # true positive
            tn = 0.0 # true negative
            fp = 0.0 # false positive
            fn = 0.0 # false negative

            # Iterate over data.
            for batch in dataiter_dict[phase]:
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    text = batch.Text
                    label = batch.Label
                    label = torch.autograd.Variable(label).long()

                    if torch.cuda.is_available():
                      text = text.cuda()
                      label = label.cuda()
                    if (batch.Text.size()[1] is not batch_size):
                      continue
                    
                    outputs = model(text)
                    outputs = F.softmax(outputs,dim=-1)                   
                    loss = criterion(outputs, label)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * text.size(0)
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == label)

                tp += torch.sum(torch.max(outputs, 1)[1] & label)
                tn += torch.sum(1-torch.max(outputs, 1)[1] & 1-label)
                fp += torch.sum(torch.max(outputs, 1)[1] & 1-label)
                fn += torch.sum(1-torch.max(outputs, 1)[1] & label)
                
            epoch_loss = running_loss / dataset_sizes[phase]
           
            sentiment_acc = float(sentiment_corrects) / dataset_sizes[phase]

            if phase == 'train':
                train_acc.append(sentiment_acc)
                train_loss.append(epoch_loss)
            elif phase == 'val':
                val_acc.append(sentiment_acc)
                val_loss.append(epoch_loss)

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} sentiment_acc: {:.4f}'.format(
                phase, sentiment_acc))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                name = str(type(model))
                torch.save(model.state_dict(), '../Models/Weights/'+ name[name.index('.')+1:-2] +'_model_test.pth')

            if phase == 'val' and epoch == num_epochs - 1:
                recall = tp / (tp + fn)
                print('recall {:.4f}'.format(recall))

        print()

    confusion_matrix = [[int(tp), int(fp)],[int(fn), int(tn)]]
    precision = tp / (tp + fp)
    f1 = 2*(precision*recall)/(precision+recall)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(float(best_loss)))

    results = {'time': time_elapsed, 
               'recall': recall,
               'precision': precision,
               'f1': f1, 
               'conf_matr': confusion_matrix,
               'val_loss': val_loss, 
               'train_loss': train_loss, 
               'val_acc': val_acc, 
               'train_acc': train_acc}
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, results

def split_and_save(text, target, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(text, target, test_size=test_size, random_state=42)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv("torchtext_data/train.csv", index=False)
    test_data.to_csv("torchtext_data/test.csv", index=False)
