import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import os
import datetime

import settings_model
import preprocess_data
from dataset_generator import create_datasets_and_loaders
from train_predict import *
from architectures.siamese_resnet import siamese_resnet18, siamese_resnet50, siamese_resnet101
from architectures.siamese_efficientnet import SiameseEfficientNet

# from utils import ContrastiveLoss, CosineSimilarityLoss

sys.path.append(os.path.join(settings_model.root_path, "rxrx1-utils"))
from rxrx import io as rio

def main():
    # preprocess data
    data = preprocess_data.rxrx_all()
    # create dataloaders
    datasets, loaders = create_datasets_and_loaders(data, settings_model.batch_size, 
                                                    settings_model.basepath_data, 
                                                    settings_model.original_image_size)
    # fix torch random seed
    seed = 42
    torch.manual_seed(seed)
    # define training parameters
    verbose = True
    model = siamese_resnet18(pretrained=True).cuda()
    # model = siamese_resnet50(pretrained=True).cuda()
    # model = SiameseEfficientNet.from_pretrained('efficientnet-b0').cuda()
    # loss_fn = ContrastiveLoss(margin=2.)
    loss_fn = CosineSimilarityLoss(margin=0.)
    lr_init = 3.e-4
    threshold = 0.5
    num_epochs = 100
    num_steps_train = 100#int(len(loaders["train"]))
    num_steps_valid = 100#int(len(loaders["valid"]))
    output_dir = os.path.join(settings_model.root_path, "models", "siamese-cell",
                              datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
    print(f"====> Outputs will be saved to {output_dir}\n")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
    # run train and evaluate
    train_and_evaluate(model, loaders["train"], loaders["valid"], 
                       lr_init, loss_fn, threshold, num_epochs, 
                       num_steps_train, num_steps_valid, settings_model.batch_size, 
                       output_dir, verbose=verbose, restore_file=None)

if __name__ == "__main__":
    main()
    
    