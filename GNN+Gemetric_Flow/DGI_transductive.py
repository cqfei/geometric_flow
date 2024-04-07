import torch
import os.path as osp
import GCL.losses as L
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index, edge_weight)
            z = act(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h=z, g=g, hn=zn)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def test2(encoder_model, data, device):
    # from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from GCL.eval.logistic_regression import LogisticRegression
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index)

    # 获取当前模型下的所有logits
    # classifier = LogisticRegression(solver='lbfgs', max_iter=5000)
    # classifier.fit(z.cpu().numpy(), data.y.cpu().numpy())
    # train_acc = classifier.score(z.cpu().numpy(), data.y.cpu().numpy())

    # y_pred = classifier.predict(z.cpu().numpy())
    # test_micro = f1_score(data.y.cpu().numpy(), y_pred, average='macro')
    # acc=accuracy_score(data.y.cpu().numpy(), y_pred)
    # print(f'Train Accuracy: {train_acc:.6f} Test F1Mi: {test_micro:.6f}, Test Accuracy: {acc:.6f}')

    x = z.detach().to(device)
    input_dim = x.size()[1]
    y = torch.Tensor(data.y).to(device)
    num_classes = y.max().item() + 1
    classifier = LogisticRegression(input_dim, num_classes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    output_fn = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss()

    best_acc = 0
    best_epoch = 0
    best_micro = 0
    best_macro = 0

    for epoch in range(5000):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(x)
        loss = criterion(output_fn(output), y)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            classifier.eval()
            y_base = y.detach().cpu().numpy()
            y_pred = classifier(x).argmax(-1).detach().cpu().numpy()
            acc = accuracy_score(y_base, y_pred)
            micf1 = f1_score(y_base, y_pred, average='micro')
            macf1 = f1_score(y_base, y_pred, average='macro')

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_micro = micf1
                best_macro = macf1
    print(
        f'Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.6f}, Best F1Mi: {best_micro:.6f}, Best F1Ma: {best_macro:.6f}')


def main():
    from data_preprossing import load_data
    # device = torch.device('cuda')
    # path = osp.join(osp.expanduser('~'), 'datasets')
    # dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    # data = dataset[0].to(device)
    device = torch.device('cuda:2')
    data = load_data('WikiCS', device)

    gconv = GConv(input_dim=data.num_features, hidden_dim=512, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=512).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=300, desc='(T)') as pbar:
        for epoch in range(1, 301):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
    test2(encoder_model, data, device)
    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
