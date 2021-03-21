import torch

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
true_y0 = torch.tensor([[2., 3.]]).to(device)
true_y01 = torch.tensor([[1., 4.]]).to(device)


batch_y = torch.stack([true_y0, true_y01], dim=0)  # (T, M, D)
print(true_y0  + 1)