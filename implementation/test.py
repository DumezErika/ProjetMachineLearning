import torch ; import torch.utils.data as data
from torchdyn.datasets import *
import torch.nn as nn
from torchdyn.models import *
import matplotlib.pyplot as plt
from litneuralode import *

def show_data(X, y):
    data = X.detach().numpy().tolist()
    couleur = []
    for i in range(len(data)):
        if y[i] > 0:
            couleur.append('plum')
        else:
            couleur.append('cornflowerblue')
    plt.scatter([i[0] for i in data],[i[1] for i in data],  color = couleur)
    plt.show()

def show_evolution(X, y):
    colors = ['cornflowerblue', 'plum']
    # compute trajectories of the trained model
    s_span = torch.linspace(0, 1, 10) ; 
    xS = nde.model.trajectory(X.to(device), s_span).detach().cpu()

    # plot resulting state-space trajectories
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(512):
        plt.plot(xS[:,i,0], xS[:,i,1], color=colors[int(y[i])], alpha=.3);
    plt.scatter(xS[-1,:,0], xS[-1,:,1], color='black', alpha=.3);
    plt.show()

def colors(data):
    colors = []
    for point in data:
        if np.linalg.norm(point) > 0.75:
            colors.append('cornflowerblue')
        else:
            colors.append('plum')
    return colors

if __name__ == "__main__":
    X, y = ToyDataset().generate(n_samples=512, noise=1e-2, dataset_type='spheres', dim=2)  
    # show_data(X, y)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train = data.TensorDataset(X.to(device), y.long().to(device))
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)

    f = nn.Sequential(DataControl(), nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))
    nde = LitNeuralODE(f).to(device)
    trainer = pl.Trainer(max_epochs=1000, progress_bar_refresh_rate=1)
    trainer.fit(nde, trainloader)

    #show_evolution(X,y)

    """
    X_test = torch.tensor(np.array([[0.5, 0], [0.8, 0]]), dtype=torch.float)
    y_test = torch.tensor(np.array([0,1,0,1]), dtype=torch.float)
    Xs_pred = nde(X_test)
    """

    X_test = torch.rand(1000,2, dtype=torch.float)
    data_test = X_test.detach().numpy().tolist()
    colors = colors(data_test)
    plt.scatter([i[0] for i in data_test],[i[1] for i in data_test], c = colors)
    plt.show()

    Xs_pred = nde(X_test)
    data = Xs_pred.detach().numpy().tolist()
    plt.scatter([i[0] for i in data],[i[1] for i in data], c = colors)
    plt.show()