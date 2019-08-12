import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import os
import datetime

import settings_model
import preprocess_data
# from dataset_generator import create_predict_datasets_and_loaders
from dataset_generator import create_siamese_datasets_and_loaders, create_predict_datasets_and_loaders
from train_predict import *
from architectures.siamese_resnet import siamese_resnet18, siamese_resnet50, siamese_resnet101
from architectures.siamese_efficientnet import SiameseEfficientNet
from architectures.siamese_densenet import siamese_densenet121

from utils import ContrastiveLoss, CosineSimilarityLoss, seed_everything

sys.path.append(os.path.join(settings_model.root_path, "rxrx1-utils"))
from rxrx import io as rio

def main(data, model_id):
    
    seed_everything(42)
    
    # create dataloaders
    datasets_siam, loaders_siam = create_siamese_datasets_and_loaders(data, settings_model.batch_size // 2, 
                                                                    settings_model.basepath_data, 
                                                                    settings_model.original_image_size)
    datasets_pred, loaders_pred = create_predict_datasets_and_loaders(data, settings_model.batch_size, 
                                                                      settings_model.basepath_data, 
                                                                      settings_model.original_image_size)

    # define training parameters
    verbose = True
    # restore_file = None
    restore_file = "/jet/prs/workspace/models/siamese-cell/HEPG2_20190811_142600/loss.best.pth.tar"
    # model = SiameseEfficientNet.from_pretrained('efficientnet-b0').cuda()
    model = siamese_densenet121(pretrained=False, embedding_size=128, drop_rate=0.3).cuda()
    split_batch = 4
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.features.conv0.parameters():
    #     param.requires_grad = True
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    
    loss_fn = CosineSimilarityLoss(margin=0.)
    lr_init = 1.e-4
    threshold = 0.99
    num_epochs = 100
    num_steps_train = int(len(loaders_siam["train"]))
    num_steps_valid = int(len(loaders_siam["valid"]))
    output_dir = os.path.join(settings_model.root_path, "models", "siamese-cell",
                              model_id + "_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
    print(f"====> Outputs will be saved to {output_dir}\n")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
    # run train and evaluate
    train_and_evaluate(model, loaders_siam["train"], loaders_siam["valid"], 
                       loaders_pred["train"], loaders_pred["valid"], 
                       lr_init, loss_fn, threshold, num_epochs, 
                       num_steps_train, num_steps_valid, settings_model.batch_size, 
                       output_dir, verbose=verbose, restore_file=restore_file, 
                       split_batch=split_batch)

def embeddings(data, model_id):
    datasets_pred, loaders_pred = create_predict_datasets_and_loaders(data, settings_model.batch_size, 
                                                                      settings_model.basepath_data, 
                                                                      settings_model.original_image_size)
    restore_file = f"/jet/prs/workspace/models/siamese-cell/{model_id}/acc.best.pth.tar"
    model = siamese_densenet121(pretrained=False, embedding_size=128).cuda()
    output_dir = os.path.join(settings_model.root_path, "models", "siamese-cell",
                              f"{model_id}", "emb")
    
    save_embeddings_for_clf(model, loaders_pred["train"], loaders_pred["valid"], output_dir)
    
if __name__ == "__main__":
    # preprocess data
    # data = preprocess_data.rxrx_all()
    # main(data, "all")
    
    # # 'HEPG2', 'HUVEC', 'RPE', 'U2OS'
    cell_type = 'HEPG2' 
    # # data = preprocess_data.rxrx_control_cell_type(cell_type)
    data = preprocess_data.rxrx_cell_type(cell_type)
    main(data, cell_type)
    # embeddings(data, "U2OS_20190812_073705")
    
    
    