# import standard libraries
import re
import time
import numpy as np
import copy
import os
import shutil
from tqdm import tqdm
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F

# import custom libs
from utils import *
import settings_model
        
def train(model, dataloader, optimizer, loss_fn, threshold, num_steps, 
          batch_size, verbose=False, save_summary_steps=5, seed=42):
    # set model to training mode
    model.train()    
    # create running averages
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()
    # create empty lists to keep track of metrics
    loss_hist_ep, loss_avg_hist_ep = [], []
    acc_hist_ep, acc_avg_hist_ep = [], []
    # initialize clocks
    start_abs = time.time()
    start = time.time()
    # iterate over batches    
    print(f'- Training on {num_steps} batches of {batch_size} images each.')
    for i, data in enumerate(dataloader):
        # break loop after num_steps batches
        if i > num_steps:
            break
        # filter soft/hard pairs
        data = filter_soft_hard_pairs(model, data)
        # extract variables    
        img_batch1, img_batch2, is_diff_batch, label_batch1, label_batch2 = data
        img_batch1 = img_batch1.type(torch.float32).cuda()
        img_batch2 = img_batch2.type(torch.float32).cuda()
        is_diff_batch = is_diff_batch.type(torch.float32).cuda()
        # compute output
        optimizer.zero_grad()
        output_batch1, output_batch2 = model(img_batch1, img_batch2)
        # compute loss
        loss = loss_fn(output_batch1, output_batch2, is_diff_batch)
        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        # update loss running average
        loss_avg.update(loss.item())
        # update loss history
        loss_hist_ep.append(loss.item())
        loss_avg_hist_ep.append(loss_avg())
        # evaluate metrics only once in a while
        if i%save_summary_steps == 0:
            # calculate similarity
            cos_f = nn.CosineSimilarity(dim=1, eps=1e-6)
            similarity = cos_f(output_batch1.detach(), output_batch2.detach())
            similarity = similarity.data.cpu().numpy()
            # get ground truth
            is_diff = is_diff_batch.data.cpu().numpy()
            # extract predictions
            prediction = (similarity < threshold).astype("int")
            # calculate accuracy
            acc = sum(prediction == is_diff) / len(prediction)
            # update accuracy running average
            acc_avg.update(acc)
            # update accuracy histories
            acc_hist_ep.append(acc)
            acc_avg_hist_ep.append(acc_avg())
            # print results every save_summary_steps if verbose=True
            if verbose:
                summary_batch_string = "loss: {:05.7f} ".format(loss.item())
                summary_batch_string += "acc: {:05.7f} ".format(acc)
                summary_batch_string += "avg loss: {:05.7f} ".format(loss_avg())
                summary_batch_string += "avg acc: {:05.7f} ".format(acc_avg())
                summary_batch_string += "{}/{:.2f} imgs/sec"\
                    .format(save_summary_steps*batch_size, time.time() - start)
                print('    It [{}/{}] '.\
                      format(i, num_steps) + summary_batch_string)
            # reset partial clock
            start = time.time()

    # log epoch summary
    summary_epoch_string = "    avg loss: {:05.7f}\n".format(loss_avg())
    summary_epoch_string += "    avg acc: {:05.7f}".format(acc_avg())
    print("- Train epoch metrics summary:\n" + summary_epoch_string)
    print('  Training run in {:.2f} minutes.'.format((time.time()-start_abs)/60.))
    
    # pack metric histories in dictionary
    histories_ep = {"loss train" : loss_hist_ep,
                    "loss avg train" : loss_avg_hist_ep,
                    "acc train" : acc_hist_ep,
                    "acc avg train" : acc_avg_hist_ep}
    
    return histories_ep

def evaluate(model, dataloader, loss_fn, threshold, num_steps, batch_size, 
             verbose=False, seed=42):
    # set model to evaluation mode
    model.eval()
    # create running averages
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()
    # initialize clock
    start = time.time()
    # iterate over batches    
    print(f'- Validating on {num_steps} batches of {batch_size} images each.')
    with torch.no_grad():
        similarity_list = []
        is_diff_list = []
        for i, data in enumerate(dataloader):
            # break loop after num_steps batches
            if i > num_steps:
                break
            # filter soft/hard pairs
            data = filter_soft_hard_pairs(model, data)
            # extract variables    
            img_batch1, img_batch2, is_diff_batch, label_batch1, label_batch2 = data
            img_batch1 = img_batch1.type(torch.float32).cuda()
            img_batch2 = img_batch2.type(torch.float32).cuda()
            is_diff_batch = is_diff_batch.type(torch.float32).cuda()
            # compute output
            output_batch1, output_batch2 = model(img_batch1, img_batch2)
            # compute loss
            loss = loss_fn(output_batch1, output_batch2, is_diff_batch)
            # update loss running average
            loss_avg.update(loss.item())
            # calculate similarity
            cos_f = nn.CosineSimilarity(dim=1, eps=1e-6)
            similarity = cos_f(output_batch1.detach(), output_batch2.detach())
            similarity = similarity.data.cpu().numpy()
            # get ground truth
            is_diff = is_diff_batch.data.cpu().numpy()
            # extract predictions
            prediction = (similarity < threshold).astype("int")
            # calculate accuracy
            acc = sum(prediction == is_diff) / len(prediction)
            # update accuracy running average
            acc_avg.update(acc)
            
            # debug threshold
            similarity_list += similarity.tolist()
            is_diff_list += is_diff.tolist()
            
        # debug threshold
        from matplotlib import pyplot as plt
        similarity = np.array(similarity_list)
        is_diff = np.array(is_diff_list)
        fig, ax = plt.subplots()
        ax.hist(similarity[is_diff==1], color="r", bins=len(similarity)//20, alpha=0.5)
        ax.hist(similarity[is_diff==0], color="g", bins=len(similarity)//20, alpha=0.5)
        plt.savefig(os.path.join(settings_model.root_path, "tmp", "dist.png"), dpi=150)
        print("- Debug threshold")
        print("  Similarity in same pairs:\n    {:.5f} +- {:.5f}"\
              .format(similarity[is_diff==0].mean(), 
                      similarity[is_diff==0].std() / len(similarity[is_diff==0])))
        print("  Similarity in different pairs:\n    {:.5f} +- {:.5f}"\
              .format(similarity[is_diff==1].mean(), 
                      similarity[is_diff==1].std() / len(similarity[is_diff==1])))
        print("  Discrepancy:\n    {:.5f}"\
              .format(-similarity[is_diff==0].mean() + similarity[is_diff==1].mean()))

    # log validation summary
    metrics_valid = {'loss' : loss_avg(),
                     'acc' : acc_avg()}
    summary_string = "    avg loss: {:05.7f}\n".format(metrics_valid["loss"])
    summary_string += "    avg acc: {:05.7f}".format(metrics_valid["acc"])
    print("- Validation metrics:\n" + summary_string)
    print('  Validation run in {:.2f} minutes.'.format((time.time()-start)/60.))

    return metrics_valid

def predict(model, dataloader_train, dataloader_predict, mode): 
    
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # iterate over dataloader_train
        for i, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            # extract variables    
            img_batch, label_batch = data
            img_batch = img_batch.type(torch.float32).cuda()
            # calculate embedding for each sample
            output_batch = model.forward_once(img_batch)
            # concatenate embeddings (and labels) in a single tensor
            if i == 0:
                embeddings_train = output_batch
                labels_train = label_batch
            else:
                embeddings_train = torch.cat((embeddings_train, output_batch), dim=0, out=None)
                labels_train += label_batch
        labels_train = np.array(labels_train)        
        
        # define similarity function
        cos_f = nn.CosineSimilarity(dim=1, eps=1e-6)
        # init accuracy
        accuracy1, accuracy2 = 0., 0.
        # iterate over all samples to predict
        for i, data in tqdm(enumerate(dataloader_predict), total=len(dataloader_predict)):
            # extract variables    
            img_batch, label_batch = data
            img_batch = img_batch.type(torch.float32).cuda()
            # calculate embedding for each sample
            output_batch = model.forward_once(img_batch)
            # compare against all training embeddings
            for embedding_predict, true_label in zip(output_batch, label_batch):
                # calculate cosine similarity
                embedding_predict_mul = embedding_predict.repeat(embeddings_train.size()[0], 1)
                similarities = cos_f(embedding_predict_mul, embeddings_train).data.cpu().numpy()
                pred_label = labels_train[np.argmax(similarities)]
                accuracy1 += (str(pred_label)==str(true_label)) / float(len(dataloader_predict.dataset))
                df = pd.DataFrame(data={"labels" : labels_train, "similarities" : similarities})
                df_scores = df.head(200).groupby("labels")["similarities"].mean().sort_values(ascending=False)
                pred_label, score = df_scores.index[0], df_scores.ix[0]
                accuracy2 += (str(pred_label)==str(true_label)) / float(len(dataloader_predict.dataset))

        print("- Accuracy (method 2) at test time:", accuracy1)
        print("- Accuracy (method 2) at test time:", accuracy2)

def save_embeddings_for_clf(model, dataloader_train, dataloader_predict, output_dir): 
    
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        with open(os.path.join(output_dir, "emb_train.csv"), "w") as f:
            # iterate over dataloader_train
            for i, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                # extract variables
                img_batch, label_batch = data
                img_batch = img_batch.type(torch.float32).cuda()
                # calculate embeddings for batch
                output_batch = model.forward_once(img_batch).data.cpu().numpy()
                # save to dataframe
                batch_size = img_batch.size()[0]
                for j in range(batch_size):
                    f.write(",".join(output_batch[j].astype("str")) +","+label_batch[j]+"\n")
        with open(os.path.join(output_dir, "emb_valid.csv"), "w") as f:
            # iterate over dataloader_predict
            for i, data in tqdm(enumerate(dataloader_predict), total=len(dataloader_predict)):
                # extract variables
                img_batch, label_batch = data
                img_batch = img_batch.type(torch.float32).cuda()
                # calculate embeddings for batch
                output_batch = model.forward_once(img_batch).data.cpu().numpy()
                # save to dataframe
                batch_size = img_batch.size()[0]
                for j in range(batch_size):
                    f.write(",".join(output_batch[j].astype("str")) +","+label_batch[j]+"\n")
                

def filter_soft_hard_pairs(model, data):
    start = time.time()
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        cos_f = nn.CosineSimilarity(dim=1, eps=1e-6)
        # extract variables    
        img_batch1, img_batch2, label_batch = data
        # initialize final tensors
        img_batch1_sh = []
        img_batch2_sh = []
        label_batch1_sh = []
        label_batch2_sh = []
        batch_size = img_batch1.size()[0]
        for i in range(batch_size):
            img = img_batch1[i]
            lbl = label_batch[i]
            # append positive pair (not the softest)
            img_batch1_sh.append(img)
            img_batch2_sh.append(img_batch2[i])
            label_batch1_sh.append(lbl)
            label_batch2_sh.append(lbl)
            # find hardest pair among the rest of batch2
            img_mul = torch.stack(batch_size * [img])
            # lbl_mul = batch_size * [lbl]
            out_mul, out_batch2 = model(img_mul.type(torch.float32).cuda(), 
                                        img_batch2.type(torch.float32).cuda())
            # compute cosine similarity
            similarity = cos_f(out_mul, out_batch2).data.cpu().numpy()
            # sort indices by similarity
            sort_idx = np.argsort(similarity)
            # find hardest pair        
            for idx in sort_idx[::-1]:
                if lbl != label_batch[idx]:
                    img_batch1_sh.append(img)
                    img_batch2_sh.append(img_batch2[idx])
                    label_batch1_sh.append(lbl)
                    label_batch2_sh.append(label_batch[idx])
                    break

        img_batch1_sh = torch.stack(img_batch1_sh)
        img_batch2_sh = torch.stack(img_batch2_sh)
        is_diff_batch_sh = torch.tensor((np.array(label_batch1_sh) == np.array(label_batch2_sh)))
            
        data = img_batch1_sh, img_batch2_sh, is_diff_batch_sh, label_batch1_sh, label_batch2_sh
    
    # print('  Soft/hard selection run in {:.2f} seconds.'.format((time.time()-start)))
        
    return data

    
def train_and_evaluate(model, dataloader_train_siam, dataloader_valid_siam, 
                       dataloader_train_pred, dataloader_valid_pred, 
                       lr_init, loss_fn, threshold,
                       num_epochs, num_steps_train, num_steps_valid, 
                       batch_size, output_dir, verbose=False, restore_file=None, seed=42):

    # load pretrained weights as initial condition
    if restore_file is not None:
        print("=> loading checkpoint {}".format(restore_file))
        checkpoint = torch.load(restore_file)
        pretrained_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_dict)
        print("=> loaded checkpoint")
    
    # initialize best validation loss and accuracy
    best_valid_loss, best_valid_acc = 1.e+15, 0.0
    # create empty list to track metrics over epochs
    loss_train_hist, loss_avg_train_hist, loss_valid_hist = [], [], []
    acc_train_hist, acc_avg_train_hist, acc_valid_hist = [], [], []
    
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init) 
    scheduler = ReduceLROnPlateau(optimizer, factor=0.9, 
                                  patience=1, cooldown=3, mode='min', verbose=True)
    if restore_file is not None:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    
    # iterate over epochs
    for epoch in range(1, num_epochs+1):
        # initialize clock       
        start = time.time()
        print("\n======================================================")
        print("Epoch [{}/{}]".format(epoch, num_epochs))
        
        # train model for a whole epoc (one full pass over the training set)
        histories_ep = train(model, dataloader_train_siam, optimizer, loss_fn, threshold, 
                             num_steps_train, batch_size, verbose=verbose)
        # update train metric histories
        loss_train_hist += histories_ep["loss train"]
        loss_avg_train_hist += histories_ep["loss avg train"]
        acc_train_hist += histories_ep["acc train"]
        acc_avg_train_hist += histories_ep["acc avg train"]
        
        # after one epoch of training, evaluate on validation set
        metrics_valid = evaluate(model, dataloader_valid_siam, loss_fn, threshold, 
                                 num_steps_valid, batch_size, verbose=verbose)
        # update train metric histories
        loss_valid_hist += len(histories_ep["loss train"]) * [metrics_valid["loss"]]
        acc_valid_hist += len(histories_ep["acc train"]) * [metrics_valid["acc"]]
        
        # update lr with scheduler
        scheduler.step(metrics_valid["loss"])
        
        # do we have a new winner?
        is_best_loss = metrics_valid["loss"]<=best_valid_loss
        is_best_acc = metrics_valid["acc"]>=best_valid_acc
        if is_best_loss:
            best_valid_loss = metrics_valid["loss"]
            print("- Found new best loss: {:.7f}".format(best_valid_loss))
        if is_best_acc:
            best_valid_acc = metrics_valid["acc"]
            print("- Found new best acc: {:.7f}".format(best_valid_acc))
            
        # try prediction
        predict(model, dataloader_train_pred, dataloader_valid_pred, mode="valid")
        
        # save checkpoints
        print("Saving checkpoints...")
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict' : optimizer.state_dict()},
                         is_best=is_best_loss,
                         metric_name='loss',
                         metric_value=metrics_valid["loss"],
                         model_dir=output_dir)
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict' : optimizer.state_dict()},
                         is_best=is_best_acc,
                         metric_name='acc',
                         metric_value=metrics_valid["acc"],
                         model_dir=output_dir)
        
        # pack global metric histories in dictionary
        print("Saving metric histories graphs...")
        histories = {"loss train" : loss_train_hist,
                     "loss avg train" : loss_avg_train_hist,
                     "loss valid" : loss_valid_hist,
                     "acc train" : acc_train_hist,
                     "acc avg train" : acc_avg_train_hist,
                     "acc valid" : acc_valid_hist
                    }
        plot_histories(histories, output_dir)
        
        # send email alert
        epoch_email_alert(output_dir)

        print('Epoch run in {:.2f} minutes'.format((time.time()-start)/60.))
