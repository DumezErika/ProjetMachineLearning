import matplotlib.pyplot as plt
import torch
import numpy as np
from dataloaders import ConcentricSphere
from dataloaders import ConcentricSquares
from torch.utils.data import DataLoader
from plots import single_feature_plt
from anode.models import ODENet
from anode.training import Trainer
from plots import get_feature_history
from plots import multi_feature_plt

if __name__ == "__main__":
    device = torch.device('cpu')

    # Création du dataset constitué de points dans un carré (les points -1), et de points dans
    # une couronne autour de ce premier carré (les points +1)
    data_concentric = ConcentricSquares(inner_range=(0., .5), outer_range=(1., 1.5), 
                                    num_points_inner=1000, num_points_outer=2000)
    dataloader = DataLoader(data_concentric, batch_size=64, shuffle=True)

    # Création de l'ODE-Net (utilisant la méthode adjointe).
    hidden_dim = 32
    model = ODENet(device, 2, hidden_dim, time_dependent=True,
                non_linearity='relu', adjoint=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, device)
    num_epochs = 18
    
    # Affiche l'évolution des points durant le training
    dataloader_viz = DataLoader(data_concentric, batch_size=256, shuffle=True)
    for inputs, targets in dataloader_viz:
        feature_history = get_feature_history(trainer, dataloader, inputs,targets, num_epochs)
        break
                                            
    multi_feature_plt(feature_history[::2], targets)

    # Affiche l'évolution de la loss durant le training
    plt.plot(trainer.histories['loss_history'])
    plt.xlim(0, len(trainer.histories['loss_history']) - 1)
    plt.ylim(0)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()