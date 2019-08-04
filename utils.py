import os
import torch
import shutil
from matplotlib import pyplot as plt
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import torch
import torch.nn.functional as F

import settings_model

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, is_diff):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-is_diff) * torch.pow(euclidean_distance, 2) +
                                      (is_diff) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def save_checkpoint(state, is_best, metric_name, metric_value, model_dir):
    """ Saves model weights and training parameters as 'last.pth.tar'. 
        If is_best==True, also saves 'metric_name.best.pth.tar'
        :param state: (dict) contains model's state_dict, 
                       may contain other keys such as epoch, optimizer state_dict
        :param is_best: (bool) true if it's the best model seen until now
        :param metric_name: name of the metric used to assess is_best
    """
    epoch = state["epoch"]
    model_name = f"epoch_{epoch}.pth.tar"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    torch.save(state, model_path)
    if is_best:
        bestmodel_path = os.path.join(model_dir, f"{metric_name}.best.pth.tar")
        shutil.copyfile(model_path, bestmodel_path)
        with open(os.path.join(model_dir, f"{metric_name}.best.txt"), "w") as f:
            f.write(f"Best model:\n  epoch: {epoch}\n  {metric_name}: {metric_value}")

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

def plot_histories(histories, output_dir):
    # save loss history plot
    plt.clf()
    plt.plot(range(len(histories['loss train'])), histories['loss train'], 
             color='k', label='loss train', alpha=0.5)
    plt.plot(range(len(histories['loss avg train'])), histories['loss avg train'], 
             color='r', ls='dashed', label='loss avg train')
    plt.plot(range(len(histories['loss valid'])), histories['loss valid'], 
             color='g', label='loss valid')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=150)
    # save accuracy history plot
    plt.clf()
    plt.plot([0, len(histories['acc train'])], [0.5,0.5], color='k', label='random benchmack')
    plt.plot(range(len(histories['acc train'])), histories['acc train'], 
             color='k', label='acc train', alpha=0.5)
    plt.plot(range(len(histories['acc avg train'])), histories['acc avg train'], 
             color='r', ls='dashed', label='acc avg train')
    plt.plot(range(len(histories['acc valid'])), histories['acc valid'], 
             color='g', label='acc valid')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "acc_history.png"), dpi=150)

def get_email_credentials():
    fpath = os.path.join(settings_model.root_path, ".email_alert.login")
    
    if os.path.exists(fpath):
        with open(fpath) as f:
            lines = f.readlines()
            sender_email = lines[0].replace("\n", "")
            password = lines[1].replace("\n", "")
            receiver_email = lines[2].replace("\n", "")
    else:
        with open(fpath, "w") as f:
            sender_email = input("Input sender email:")
            password = input("Input sender password:")
            receiver_email = input("Input receiver email:")
            f.write(f"{sender_email}\n{password}\n{receiver_email}")
            
    return sender_email, password, receiver_email
    
def epoch_email_alert(output_dir):
    port = 465
    sender_email, password, receiver_email = get_email_credentials()

    subject = "Recursion Kaggle model: new epoch completed!"
    body = "This is an email with attachment sent from Python."

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    attachment1_path = os.path.join(output_dir, "loss_history.png")
    attachment2_path = os.path.join(output_dir, "acc_history.png")
    attachment3_path = os.path.join(settings_model.root_path, "tmp", "eucl_dist.png")
    
    for attachment_path in [attachment1_path, attachment2_path, attachment3_path]:
        # Open PDF file in binary mode
        with open(attachment_path, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {attachment_path}",
        )

        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)        

        