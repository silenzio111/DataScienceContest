import torch
import torch.nn.functional as F
from preprocess import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=VAE().to(device)
model.load_state_dict(torch.load('model/vae_model_1.0810.pth'))

