import torch
import torch.nn.functional as F

#加载GCN, GAT, SAGE,GAE, VGAE, DGI,GMI,MVGRL,GCA, GraphCL, GRACE卷积层
# from torch_geometric.nn.conv.gcn_conv import GCNConv
# from torch_geometric.nn.conv.gat_conv import GATConv
# from torch_geometric.nn.conv.sage_conv import SAGEConv

# from torch_geometric.nn.conv.gin_conv import GINCon
from torch_geometric.nn import GCN, GAE,VGAE,GAT,GIN,GraphSAGE
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv,SAGEConv,GINConv
import torch.nn as nn
# 定义模型
class GCN_NN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()
        # 根据conv_type参数，选择对应的卷积层。由于框架的统一，不同卷积层有相同的初始化参数设置和接收的数据格式。

        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        # self.gcn=GCN(feature_dim, hidden_dim,out_dim)
        # 使用Glorot初始化来初始化权重
        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        #第二层是否需要加权重？可以测试下
        logits = self.conv2(h, edge_index, edge_weight=edge_weight)
        # logits = self.conv2(h, edge_index)
        return logits
        # return self.gcn(x, edge_index, edge_weight=edge_weight)
class GAT_NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(GAT_NN, self).__init__()
        self.gat = GAT(input_dim, hidden_dim,out_channels=output_dim, heads=num_heads, num_layers=num_layers)

    def forward(self, x, edge_index, edge_weight=None):
        # return self.gat(x, edge_index, edge_weight=edge_weight)

        return self.gat(x, edge_index,edge_attr=edge_weight)


    # def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
    #     super(GAT_NN, self).__init__()
    #
    #
    #     # 第一层 GAT
    #     self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
    #     # 第二层 GAT
    #     self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=2)
    #
    #     # 使用 Glorot 初始化来初始化权重
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     # 使用 Glorot 初始化来初始化权重
    #     self.conv1.reset_parameters()
    #     self.conv1.reset_parameters()
    #
    # def forward(self, x, edge_index, edge_weight=None):
    #     # x, edge_index = data.x, data.edge_index
    #
    #     # 第一层 GAT
    #     x = self.conv1(x, edge_index)
    #     x = torch.relu(x)
    #     if edge_weight is not None:
    #         # 将 edge_weight 与注意力分数相乘
    #         # x = x * edge_weight.view(-1, 1)
    #         x=torch.matmul(edge_weight, x)
    #     # 第二层 GAT
    #     x = self.conv2(x, edge_index)
    #
    #     return x


class GIN_NN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super(GIN_NN, self).__init__()
        self.gin = GIN(feature_dim, hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        return self.gin(x, edge_index, edge_weight=edge_weight)


class SAGE_NN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super(SAGE_NN,self).__init__()
        self.sage=GraphSAGE(feature_dim, hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        return self.sage(x, edge_index, edge_weight=edge_weight)
# class GAE_NN(torch.nn.Module):\
#     #这样写是错误的，下面的写法本质还是GCN
#     def __init__(self, input_dim, hidden_dim,out_dim):
#         super(GAE_NN, self).__init__()
#         self.encoder = GCN(input_dim, hidden_dim)
#         self.decoder = GCN(hidden_dim, out_dim)
#         # self.gae= GAE(self.encoder,self.decoder)
#         # self.gae = GAE(encoder=GCN(input_dim, hidden_dim),
#         #                decoder=GCN(hidden_dim, out_dim))
#
#
#     def forward(self, x, edge_index, edge_weight=None):
#         # 只在encoder加权重
#         z = self.encoder(x, edge_index,edge_weight=edge_weight).relu()
#         # # 解码器
#         recon_x = self.decoder(z, edge_index)
#
#         return recon_x

class GAE_NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
            super(GAE_NN, self).__init__()
            self.encoder = GCNConv(input_dim, hidden_dim)
            self.decoder = GCNConv(hidden_dim, input_dim)
            self.gae= GAE(self.encoder,self.decoder)

    def forward(self, x, edge_index, edge_weight=None):
            # 只在encoder加权重
            z = self.encoder(x, edge_index,edge_weight=edge_weight).relu()
            # # 解码器
            recon_x = self.decoder(z, edge_index)

            return z,recon_x

class VGAE_NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAE_NN, self).__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.mu_layer = GCNConv(hidden_dim, latent_dim)
        self.logvar_layer = GCNConv(hidden_dim, latent_dim)
        self.decoder = GCNConv(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index,edge_weight=edge_weight)
        mu = self.mu_layer(z, edge_index)
        logvar = self.logvar_layer(z, edge_index)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z, edge_index)
        return z, reconstructed_x, mu, logvar


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))