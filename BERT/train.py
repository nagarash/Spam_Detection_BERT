import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from transformers import AutoModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW
from model import BERT_Arch

writer = SummaryWriter('/opt/ml/output/tensorboard/')

def model_fn():
    """Load the PyTorch model"""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)
    
    
    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False
    
    # push the model to GPU
    model.to(device)

    print("Done loading model.")
    return model

def _get_train_dataset(training_dir):
    print("Get train dataset.")
      
    # read tensors from pickled object
    with open("{}/train_tensors.pkl".format(training_dir), "rb") as f:
        train_seq, train_mask, train_y = pickle.load(f)
        
    # wrap tensors
    train_yt = torch.tensor(train_y.tolist())
    train_data = TensorDataset(train_seq, train_mask, train_yt)
    
    return train_data

def _get_dataloader(dataset, batch_size):
    print("Get train data loader.")
    
    # sampler for sampling the data during training
    sampler = RandomSampler(dataset)

    # dataLoader for train set
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return dataloader
    
    
def crossvalidate(model, train_dataset, epochs, batch_size, optimizer, loss_fn, device, k_fold=5):
    
    train_scores = []
    val_scores = []
    
    train_cv_preds = []
    val_cv_preds = []
    
    total_size = len(train_dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        # train the model
        train_set = torch.utils.data.dataset.Subset(train_dataset,train_indices)
        train_dataloader = _get_dataloader(train_set, batch_size)
        train_acc, train_preds = train(i, model, train_dataloader, epochs, optimizer, loss_fn, device)
        train_scores.append(train_acc)
        train_cv_preds.append(train_preds)
        
        # validate
        val_set = torch.utils.data.dataset.Subset(train_dataset,val_indices)
        val_dataloader = _get_dataloader(val_set, batch_size)
        val_acc, val_preds = evaluate(i, model, val_dataloader, optimizer, loss_fn, device)
        val_scores.append(val_acc)
        val_cv_preds.append(val_preds)
    
    return train_scores,val_scores,train_cv_preds,val_cv_preds

def train(fold, model, train_dataloader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
  
    model.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = loss_fn(preds, labels)
        
        writer.add_scalar('training_loss', loss.item(), step, fold)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

def evaluate(fold, model, val_dataloader, optimizer, loss_fn, device):
    print("\nEvaluating...")
    
    # deactivate dropout layers
    model.eval()

    total_loss = 0
  
    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):
        
        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = loss_fn(preds,labels)
            writer.add_scalar('validation_loss', loss.item(), step, fold)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    
    return avg_loss, total_preds


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--k_fold', type=int, default=5, metavar='N',
                        help='number of folds for cross-validation (default: 5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')


    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_dataset = _get_train_dataset(args.data_dir)

    # push the model to GPU
    model = model_fn()
    
    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = 1e-5) 
    
    # define the loss function
    loss_fn  = torch.nn.NLLLoss() 
    
    
    # Train the model.
    train_loss, val_loss, train_cv_preds, val_cv_preds = crossvalidate(model, train_dataset, 
                                                                       args.epochs, args.batch_size, optimizer, 
                                                                       loss_fn, device, args.k_fold)


    # Save the model parameters
    loss_path = os.path.join(args.model_dir, 'losses.pt')
    with open(loss_path, 'wb') as f:
        pickle.dump([train_loss, val_loss, train_cv_preds, val_cv_preds], f)
    
    
    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pt')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
