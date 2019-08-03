import os
import torch
import shutil

def save_checkpoint(state, is_best, epoch, metric_name, model_dir):
    """ Saves model weights and training parameters as 'last.pth.tar'. 
        If is_best==True, also saves 'metric_name.best.pth.tar'
        Args:
              state: (dict) contains model's state_dict, 
              may contain other keys such as epoch, optimizer state_dict
              is_best: (bool) True if it is the best model seen till now
              epoch: epoch number
              metric_name: name of the metric used to assess is_best
    """
    model_name = f"epoch{epoch}.pth.tar"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, model_name)
    torch.save(state, model_path)
    if is_best:
        bestmodel_path = os.path.join(model_dir, metric_name+".best.pth.tar")
        shutil.copyfile(model_path, bestmodel_path) 

class RunningAverage():
    """ A simple class that maintains the running average of a quantity
        Example:
            loss_avg = RunningAverage()
            loss_avg.update(2)
            loss_avg.update(4)
            loss_avg() = 3
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

def plot_loss_history(histories, output_dir):
    plt.clf()
    plt.plot(range(len(histories['loss train'])), histories['loss train'], 
             color='k', label='loss train')
    plt.plot(range(len(histories['loss avg train'])), histories['loss avg train'], 
             color='r', ls='dashed', label='loss avg train')
    plt.plot(range(len(histories['loss validation'])), histories['loss validation'], 
             color='g', label='loss validation')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=150)
