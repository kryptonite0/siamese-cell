# import standard libraries
import re
import time
import numpy as np
import copy
import os
import shutil

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
    # fix random seed
    torch.manual_seed(seed)
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
            # calculate distance
            cos_f = nn.CosineSimilarity(dim=1, eps=1e-6)
            distance = cos_f(output_batch1.detach(), output_batch2.detach())
            distance = distance.data.cpu().numpy()
            # distance = (output_batch1.detach() - output_batch2.detach()).pow(2).sum(1)
            # distance = distance.data.cpu().numpy()
            # get ground truth
            is_diff = is_diff_batch.data.cpu().numpy()
            # extract predictions
            prediction = (distance < threshold).astype("int")
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
    # fix random seed
    torch.manual_seed(seed)
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
        distance_list = []
        is_diff_list = []
        for i, data in enumerate(dataloader):
            # break loop after num_steps batches
            if i > num_steps:
                break
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
            # calculate distance
            cos_f = nn.CosineSimilarity(dim=1, eps=1e-6)
            distance = cos_f(output_batch1.detach(), output_batch2.detach())
            distance = distance.data.cpu().numpy()
            # get ground truth
            is_diff = is_diff_batch.data.cpu().numpy()
            # extract predictions
            prediction = (distance < threshold).astype("int")
            # calculate accuracy
            acc = sum(prediction == is_diff) / len(prediction)
            # update accuracy running average
            acc_avg.update(acc)
            
            # debug threshold
            distance_list += distance.tolist()
            is_diff_list += is_diff.tolist()
            
        # debug threshold
        from matplotlib import pyplot as plt
        distance = np.array(distance_list)
        is_diff = np.array(is_diff_list)
        fig, ax = plt.subplots()
        ax.hist(distance[is_diff==1], color="r", bins=200, alpha=0.5)
        ax.hist(distance[is_diff==0], color="g", bins=200, alpha=0.5)
        plt.savefig(os.path.join(settings_model.root_path, "tmp", "dist.png"), dpi=150)
        print("- Debug threshold")
        print("  Distance in same pairs:\n    {:.5f} +- {:.5f}"\
              .format(distance[is_diff==0].mean(), 
                      distance[is_diff==0].std() / len(distance[is_diff==0])))
        print("  Distance in different pairs:\n    {:.5f} +- {:.5f}"\
              .format(distance[is_diff==1].mean(), 
                      distance[is_diff==1].std() / len(distance[is_diff==1])))
        print("  Discrepancy:\n    {:.5f}"\
              .format(-distance[is_diff==0].mean() + distance[is_diff==1].mean()))

    # log validation summary
    metrics_valid = {'loss' : loss_avg(),
                     'acc' : acc_avg()}
    summary_string = "    avg loss: {:05.7f}\n".format(metrics_valid["loss"])
    summary_string += "    avg acc: {:05.7f}".format(metrics_valid["acc"])
    print("- Validation metrics:\n" + summary_string)
    print('  Validation run in {:.2f} minutes.'.format((time.time()-start)/60.))

    return metrics_valid

# def predict_valid(model, dataloader): 

#     # set model to evaluation mode
#     model.eval()
    
#     zslices = []
#     ztargets = []
#     zpreds = []
#     with torch.no_grad():
#         for i, (input_batch, labels_batch) in enumerate(dataloader):
#             print('Predicting batch {} / {}.'.format(i+1, len(dataloader)))
#             # Send torch tensor to cuda
#             input_batch = input_batch.cuda(async=True)
#             # compute output
#             output_batch = model(input_batch.cuda())
#             # sigmoid the output
#             sig = nn.Sigmoid().cuda()
#             output_batch = sig(output_batch)
#             # extract numpy arrays
#             input_batch = input_batch.data.cpu().numpy()
#             labels_batch = labels_batch.data.cpu().numpy()
#             output_batch = output_batch.data.cpu().numpy()
#             # iterate and append to lists
#             for img, trg, pred in zip(input_batch, labels_batch, output_batch):
#                 zslices.append(img[1]) # 2nd channel corresponds to middle zslice
#                 ztargets.append(trg[0]) # channel 0 (there's only one)
#                 zpreds.append(pred[0]) # channel 0 (there's only one)
#     # save into dataframe
#     df_pred = pd.DataFrame(data={"zslice" : zslices, 
#                                  "ztarget" : ztargets, 
#                                  "zpred" : zpreds})        
#     return df_pred

# def predict_test(model, dataloader, output_dir, apply_lung_mask, sample_name="test"): 
#     # set model to evaluation mode
#     model.eval()
#     with torch.no_grad():
#         start = time.time()
#         for i, (input_batch, mask_batch, img_id_batch) in enumerate(dataloader):
#             # Send torch tensor to cuda
#             input_batch = input_batch.cuda(async=True)
#             # compute output
#             output_batch = model(input_batch)
#             sig = nn.Sigmoid().cuda()
#             output_batch = sig(output_batch)
#             # extract numpy arrays
#             output_batch = output_batch.data.cpu().numpy()
#             mask_batch = mask_batch.numpy()
#             # save to png
#             for img_id, output, mask in zip(img_id_batch, output_batch, mask_batch):
#                 seriesuid = img_id.split("_")[0]
#                 slice_n = img_id.split("_")[1]
#                 ct_pred_dir = os.path.join(output_dir, f"{sample_name}_predictions", seriesuid)
#                 if not os.path.exists(ct_pred_dir):
#                     os.makedirs(ct_pred_dir, exist_ok=True)
#                 zpred = output[0]
#                 if apply_lung_mask:
#                     zmask = mask[0]
#                     zpred[zmask==0.] = 0.
#                 cv2.imwrite(os.path.join(ct_pred_dir, "zpred_" + str(slice_n).rjust(4, '0') + ".png"), 
#                             zpred * 255)
#             print('Predicted batch {} / {} in {:.1f} seconds.'\
#                   .format(i+1, len(dataloader), time.time() - start))
#             start = time.time()

# def predict_scan_for_production(model, dataloader, output_dir, apply_lung_mask):
#     # create empty prediction cube
#     vxl_array_pred = []
#     # set model to evaluation mode
#     model.eval()
#     with torch.no_grad():
#         start = time.time()
#         for i, (input_batch, mask_batch, img_id_batch) in enumerate(dataloader):
#             # Send torch tensor to cuda
#             input_batch = input_batch.cuda(async=True)
#             # compute output
#             output_batch = model(input_batch)
#             sig = nn.Sigmoid().cuda()
#             output_batch = sig(output_batch)
#             # extract numpy arrays
#             output_batch = output_batch.data.cpu().numpy()
#             mask_batch = mask_batch.numpy()
#             # save to png
#             for img_id, output, mask in zip(img_id_batch, output_batch, mask_batch):
#                 seriesuid = img_id.split("_")[0]
#                 slice_n = img_id.split("_")[1]
#                 zpred = output[0]
#                 if apply_lung_mask:
#                     zmask = mask[0]
#                     # dilate mask to include lung borders
#                     zmask = binary_dilation(zmask, iterations=3)
#                     zpred[zmask==0.] = 0.
#                 # append prediction slice to cube
#                 vxl_array_pred.append(zpred)
#             print('Predicted batch {} / {} in {:.1f} seconds.'\
#                   .format(i+1, len(dataloader), time.time() - start))
#             start = time.time()
            
#         return np.asarray(vxl_array_pred)
    
def train_and_evaluate(model, dataloader_train, dataloader_valid, lr_init, loss_fn, threshold,
                       num_epochs, num_steps_train, num_steps_valid, 
                       batch_size, output_dir, verbose=False, restore_file=None, seed=42):
    # fix random seed
    torch.manual_seed(seed)
    
    # # load pretrained weights as initial condition
    # if restore_file is not None:
    #     # reload densenet weights from restore_file if specified
    #     print("=> loading checkpoint {}".format(restore_file))
    #     checkpoint = torch.load(restore_file)
    #     # pattern addresses this issue https://github.com/KaiyangZhou/deep-person-reid/issues/23
    #     pattern = re.compile(
    #         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    #     pretrained_dict = checkpoint['state_dict']
    #     for key in list(pretrained_dict.keys()):
    #         res = pattern.match(key)
    #         if res:
    #             new_key = res.group(1) + res.group(2)
    #             pretrained_dict[new_key] = pretrained_dict[key]
    #             del pretrained_dict[key]
    #     # get model weights dict (empty)
    #     model_dict = model.state_dict()
    #     # align dictionaries
    #     for key in model_dict.keys():
    #         if key in pretrained_dict:
    #             model_dict[key] = pretrained_dict[key]
    #     model.load_state_dict(model_dict)
    #     print("=> loaded checkpoint")
    
    # initialize best validation loss and accuracy
    best_valid_loss, best_valid_acc = 1.e+15, 0.0
    # create empty list to track metrics over epochs
    loss_train_hist, loss_avg_train_hist, loss_valid_hist = [], [], []
    acc_train_hist, acc_avg_train_hist, acc_valid_hist = [], [], []
    
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init) 
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, cooldown=2, mode='min', verbose=True)
#     if restore_file is not None:
#         optimizer.load_state_dict(checkpoint['optimizer'])
    
    # iterate over epochs
    for epoch in range(1, num_epochs+1):
        # initialize clock       
        start = time.time()
        print("\n======================================================")
        print("Epoch [{}/{}]".format(epoch, num_epochs))
        
        # train model for a whole epoc (one full pass over the training set)
        histories_ep = train(model, dataloader_train, optimizer, loss_fn, threshold, 
                             num_steps_train, batch_size, verbose=verbose)
        # update train metric histories
        loss_train_hist += histories_ep["loss train"]
        loss_avg_train_hist += histories_ep["loss avg train"]
        acc_train_hist += histories_ep["acc train"]
        acc_avg_train_hist += histories_ep["acc avg train"]
        
        # after one epoch of training, evaluate on validation set
        metrics_valid = evaluate(model, dataloader_valid, loss_fn, threshold, 
                                 num_steps_valid, batch_size, verbose=verbose)
        # update train metric histories
        loss_valid_hist += len(histories_ep["loss train"]) * [metrics_valid["loss"]]
        acc_valid_hist += len(histories_ep["acc train"]) * [metrics_valid["acc"]]
        
        # update lr with scheduler
        # scheduler.step(metrics_valid["loss"])
        
        # do we have a new winner?
        is_best_loss = metrics_valid["loss"]<=best_valid_loss
        is_best_acc = metrics_valid["acc"]>=best_valid_acc
        if is_best_loss:
            best_valid_loss = metrics_valid["loss"]
            print("- Found new best loss: {:.7f}".format(best_valid_loss))
        if is_best_acc:
            best_valid_acc = metrics_valid["acc"]
            print("- Found new best acc: {:.7f}".format(best_valid_acc))
            
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
