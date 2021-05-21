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

def get_square_aspect_ratio(plt_axis):
    return np.diff(plt_axis.get_xlim())[0] / np.diff(plt_axis.get_ylim())[0]
    
if __name__ == "__main__":
    device = torch.device('cpu')

    data_dim = 2
    data_concentric = ConcentricSphere(data_dim, inner_range=(0., .5), outer_range=(1., 1.5), 
                                    num_points_inner=1000, num_points_outer=2000)
    dataloader = DataLoader(data_concentric, batch_size=64, shuffle=True)

    hidden_dim = 32

    model = ODENet(device, data_dim, hidden_dim, time_dependent=True,
                non_linearity='relu', adjoint=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model, optimizer, device)
    num_epochs = 12
    trainer.train(dataloader, num_epochs)

    num_points = 200
    x1s = np.linspace(-2.0, 2.0, num_points)
    x2s = np.linspace(-2.0, 2.0, num_points)

    points = torch.tensor(np.transpose([np.tile(x1s, len(x2s)), np.repeat(x2s, len(x1s))]), dtype=torch.float32)
    Y = model.forward(points).reshape(num_points, num_points).detach().numpy()

    X1, X2 = np.meshgrid(x1s, x2s)

    plt.pcolormesh(X1, X2, Y, cmap=plt.cm.get_cmap('YlGn'))
    plt.colorbar()

    dataloader_viz = DataLoader(data_concentric, batch_size=3000, shuffle=True)
    for inputs, targets in dataloader_viz:
        alpha = 0.5
        color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
        num_dims = inputs.shape[1]
        plt.scatter(inputs[:, 0].numpy(), inputs[:, 1].numpy(), c=color,
                        alpha=alpha, linewidths=0)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                            labelbottom=False, right=False, left=False,
                            labelleft=False)

    plt.show()