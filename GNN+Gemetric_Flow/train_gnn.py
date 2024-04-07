import os
import re,json,codecs
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import GCL.losses as L
import GCL.augmentors as A
import time
from log import setup_logger,change_log_file
logger = setup_logger('my_logger', f'./logs/log.log')

import matplotlib.pyplot as plt
from models import GCN_NN,GAT_NN,GAE_NN
from DGI_transductive import GConv as DGI_GConv,Encoder as DGI_Encoder
from GCL.models import SingleBranchContrast,DualBranchContrast
from GRACE import GConv as GRACE_GConv,Encoder as GRACE_Encoder

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from GCL.eval.logistic_regression import LogisticRegression as LogisticRegression_GCL

from data_preprossing import load_data, load_split_mask,load_created_data

args_all=json.load(codecs.open('./config.json', 'r', 'utf-8'))

model_name=args_all['model_names'][args_all["choices"]["model_name"]]

device=args_all['devices'][args_all["choices"]["device"]]
epoch_step=args_all["choices"]["epoch_step"]
logger.info(f'model: {model_name}, device: {device}')

import warnings
warnings.filterwarnings("ignore", message="'dropout_adj' is deprecated")

def compute_f1(logits, label):
    prediction = logits.max(dim=-1)[1]
    f1 = f1_score(label.cpu().numpy(), prediction.cpu().numpy(), average='micro')
    return f1

def train(data,dataset_name,model,best_model_dir,train_mask,val_mask,add_edge_weight=False,model_args=None):
    learning_rate = model_args[model_name][dataset_name]['learning_rate']
    weight_decay = model_args[model_name][dataset_name]['weight_decay']
    epochs = model_args[model_name][dataset_name]['epochs']
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #################################################
    loss_history = []
    train_f1_history = []
    val_f1_history = []
    best_val_f1 = 0
    best_epoch=0
    for i in range(epochs):
        if add_edge_weight:
            logits = model(data.x, data.edge_index, data.edge_weight)
        else:
            logits = model(data.x, data.edge_index, edge_weight=None)

        train_logits = logits[train_mask]
        train_y = data.y[train_mask]
        loss = criterion(train_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_f1 = compute_f1(train_logits, train_y)

        val_logits, val_label = logits[val_mask], data.y[val_mask]
        val_f1 = compute_f1(val_logits, val_label)

        loss_history.append(loss.item())
        train_f1_history.append(train_f1.item())
        val_f1_history.append(val_f1.item())

        if i % epoch_step == 0:
           logger.info("Epoch: {:03d}: Loss {:.6f}, Trainf1 {:.6f}, Valf1 {:.6f}".format(i, loss.item(), train_f1.item(),
                                                                                 val_f1.item()))
        if best_val_f1<val_f1.item():
            best_val_f1=val_f1.item()
            torch.save(model.state_dict(), best_model_dir)
            best_epoch=i
    logger.info(f'Best Epoch: {best_epoch}')
    return loss_history,train_f1_history,val_f1_history,best_epoch

def train_GAE(data,dataset_name,model,best_model_dir,train_mask,val_mask,add_edge_weight=False,model_args=None):
    learning_rate = model_args[model_name][dataset_name]['learning_rate']
    weight_decay = model_args[model_name][dataset_name]['weight_decay']
    epochs = model_args[model_name][dataset_name]['epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    #################################################
    loss_history = []
    train_f1_history = []
    val_f1_history = []
    # train_ap_history = []
    # val_ap_history = []
    # train_auc_history = []
    # val_auc_history = []
    best_val_f1 = 0
    best_epoch=0
    model.train()
    for epoch in range(epochs):

        optimizer.zero_grad()
        if model_name=='GAE':
            if add_edge_weight:
                z,recon_x = model(data.x, data.edge_index,data.edge_weight)
            else:
                z,recon_x = model(data.x, data.edge_index)
            loss = torch.nn.functional.mse_loss(recon_x[train_mask], data.x[train_mask])
        elif model_name=='VGAE':
            if add_edge_weight:
                z, reconstructed_x, mu, logvar = model(data.x, data.edge_index,data.edge_weight)
            else:
                z, reconstructed_x, mu, logvar = model(data.x, data.edge_index)
            loss = torch.nn.functional.mse_loss(reconstructed_x[train_mask], data.x[train_mask])
            kl_loss = -0.5 * torch.sum(1 + logvar[train_mask] - mu[train_mask].pow(2) - logvar[train_mask].exp())
            loss += kl_loss

        loss.backward()
        optimizer.step()

        if epoch % epoch_step == 0:
            with torch.no_grad():
                classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
                classifier.fit(z[train_mask].cpu().numpy(), data.y[train_mask].cpu().numpy())
                train_f1 = classifier.score(z[train_mask].cpu().numpy(), data.y[train_mask].cpu().numpy())
                y_val_pred = classifier.predict(z[val_mask].cpu().numpy())
                val_f1 = f1_score(data.y[val_mask].cpu().numpy(), y_val_pred, average='micro')
                # logger.info("Epoch: {:03d}: Loss {:.4f}, Trainf1 {:.4f}, Valf1 {:.4f}".format(epoch, loss.item(),
                #                                                                                 train_f1.item(),
                #                                                                                 val_f1.item()))
        loss_history.append(loss.item())
        train_f1_history.append(train_f1)
        val_f1_history.append(val_f1)

        if best_val_f1<val_f1:
            best_val_f1=val_f1
            torch.save(model.state_dict(), best_model_dir)
            best_epoch=epoch
    logger.info(f'Best Epoch: {best_epoch}')
    return loss_history,train_f1_history,val_f1_history,best_epoch

def train_GCL(data,dataset_name, encoder_model,contrast_model,best_model_dir,train_mask,val_mask,add_edge_weight=False,model_args=None):
    learning_rate = model_args[model_name][dataset_name]['learning_rate']
    weight_decay = model_args[model_name][dataset_name]['weight_decay']
    epochs = model_args[model_name][dataset_name]['epochs']
    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #################################################
    loss_history = []
    train_f1_history = []
    val_f1_history = []

    best_val_f1 = 0
    best_epoch = 0
    encoder_model.train()
    for epoch in range(epochs):

        encoder_model.train()
        optimizer.zero_grad()
        if model_name=='DGI':
            if add_edge_weight:
                z, g, zn = encoder_model(data.x, data.edge_index,data.edge_weight)
            else:
                z, g, zn = encoder_model(data.x, data.edge_index)
            loss = contrast_model(h=z, g=g, hn=zn)
        elif model_name=='GRACE':
            if add_edge_weight:
                z, z1, z2 = encoder_model(data.x, data.edge_index,data.edge_weight)
            else:
                z, z1, z2 = encoder_model(data.x, data.edge_index)
            h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
            loss = contrast_model(h1, h2)

        loss.backward()
        optimizer.step()

        if epoch % epoch_step == 0:
            x = z.detach().to(device)
            y = torch.Tensor(data.y).to(device)
            classifier=train_LR(x,y,train_mask,val_mask)

            y_train_pred = classifier(x[train_mask]).argmax(-1).detach().cpu().numpy()
            train_f1 = f1_score(data.y[train_mask].cpu().numpy(), y_train_pred, average='micro')

            y_val_pred = classifier(x[val_mask]).argmax(-1).detach().cpu().numpy()
            val_f1 = f1_score(data.y[val_mask].cpu().numpy(), y_val_pred, average='micro')

            logger.info("Epoch: {:03d}: Loss {:.4f}, Trainf1 {:.4f}, Valf1 {:.4f}".format(epoch, loss.item(),
                                                                                                train_f1.item(),
                                                                                                val_f1.item()))
        loss_history.append(loss.item())
        train_f1_history.append(train_f1)
        val_f1_history.append(val_f1)

        if best_val_f1 < val_f1:
            best_val_f1 = val_f1
            torch.save(encoder_model.state_dict(), best_model_dir)
            best_epoch = epoch
    logger.info(f'Best Epoch: {best_epoch}')

    return loss_history, train_f1_history, val_f1_history, best_epoch

def test(data,model,best_model_dir,train_mask,val_mask,test_mask,add_edge_weight=False):
    model.load_state_dict(torch.load(best_model_dir,next(model.parameters()).device))
    model.eval()
    with torch.no_grad():
        if add_edge_weight:
            logits = model(data.x, data.edge_index,data.edge_weight)
        else:
            logits = model(data.x, data.edge_index)

        train_logits, train_label = logits[train_mask], data.y[train_mask]
        train_f1 = compute_f1(train_logits, train_label)
        logger.info("Best Train f1: {:.6f}".format(train_f1.item()))

        val_logits, val_label = logits[val_mask], data.y[val_mask]
        val_f1 = compute_f1(val_logits, val_label)
        logger.info("Best Val f1: {:.6f}".format(val_f1.item()))

        test_logits, test_label = logits[test_mask], data.y[test_mask]
        test_f1 = compute_f1(test_logits, test_label)
        logger.info("Best Test f1: {:.6f}".format(test_f1.item()))
    return test_f1.item()

def test_GAE(data,model,best_model_dir,train_mask,val_mask,test_mask,add_edge_weight=False):
    model.load_state_dict(torch.load(best_model_dir, next(model.parameters()).device))
    model.eval()
    if model_name=='GAE':
        if add_edge_weight:
            z ,x_recon= model(data.x, data.edge_index, data.edge_weight)
        else:
            z, x_recon = model(data.x, data.edge_index)
    elif model_name=='VGAE':
        if add_edge_weight:
            z, reconstructed_x, mu, logvar = model(data.x, data.edge_index,data.edge_weight)
        else:
            z, reconstructed_x, mu, logvar = model(data.x, data.edge_index)

    with torch.no_grad():
        classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
        classifier.fit(z[train_mask].cpu().numpy(), data.y[train_mask].cpu().numpy())
        train_f1 = classifier.score(z[train_mask].cpu().numpy(), data.y[train_mask].cpu().numpy())
        logger.info(f'Train f1: {train_f1:.6f}')

        y_val_pred = classifier.predict(z[val_mask].cpu().numpy())
        val_f1 = f1_score(data.y[val_mask].cpu().numpy(), y_val_pred, average='micro')
        logger.info(f'Validation f1: {val_f1:.6f}')

        y_test_pred = classifier.predict(z[test_mask].cpu().numpy())
        test_f1 = f1_score(data.y[test_mask].cpu().numpy(), y_test_pred, average='micro')
        logger.info(f'Test f1: {test_f1:.6f}')

    return test_f1

def test_GCL(data,dataset_name,encoder_model,contrast_model,best_model_dir,train_mask,val_mask,test_mask,add_edge_weight=False,model_args=None):

    encoder_model.load_state_dict(torch.load(best_model_dir, next(encoder_model.parameters()).device))
    encoder_model.eval()
    if add_edge_weight:
        z = encoder_model(data.x, data.edge_index, data.edge_weight)
    else:
        z, _, _ = encoder_model(data.x, data.edge_index)

    x = z.detach().to(device)
    y = torch.Tensor(data.y).to(device)
    classifier = train_LR(x, y, train_mask, val_mask)
    y_train_pred = classifier(x[train_mask]).argmax(-1).detach().cpu().numpy()
    train_f1 = f1_score(data.y[train_mask].cpu().numpy(), y_train_pred, average='micro')
    logger.info(f'Train f1: {train_f1:.6f}')

    y_val_pred = classifier(x[val_mask]).argmax(-1).detach().cpu().numpy()
    val_f1 = f1_score(data.y[val_mask].cpu().numpy(), y_val_pred, average='micro')
    logger.info(f'Validation f1: {val_f1:.6f}')
    
    y_test_pred = classifier(x[test_mask]).argmax(-1).detach().cpu().numpy()
    test_f1 = f1_score(data.y[test_mask].cpu().numpy(), y_test_pred, average='micro')
    logger.info(f'Test f1: {test_f1:.6f}')

    return test_f1

def train_LR(x,y,train_mask,val_mask):
    input_dim = x.size()[1]
    num_classes = y.max().item() + 1
    classifier = LogisticRegression_GCL(input_dim, num_classes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    output_fn = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss()

    best_f1 = 0
    best_epoch = 0
    best_LR=None
    for epoch in range(3000):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(x)
        loss = criterion(output_fn(output[train_mask]), y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            classifier.eval()
            y_test = y[val_mask].detach().cpu().numpy()
            y_pred = classifier(x[val_mask]).argmax(-1).detach().cpu().numpy()
            f1=f1_score(y_test, y_pred, average='micro')

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                best_LR_classifier=classifier
    return best_LR_classifier

def plot_loss_f1(loss_history,train_f1_history,val_f1_history,best_model_dir,i):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label='Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Cross {i} Training Loss')

    plt.subplot(1, 3, 2)
    plt.plot(train_f1_history, label='f1', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('f1')
    plt.title(f'Cross {i} Train f1')

    plt.subplot(1, 3, 3)
    plt.plot(val_f1_history, label='f1', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('f1')
    plt.title(f'Cross {i} Val f1')

    plt.tight_layout()
    file_name = re.findall('./models/(.*?)_cross', best_model_dir)
    if not os.path.exists(f'./results/figures/{file_name[0]}'):
        os.mkdir(f'./results/figures/{file_name[0]}')
    img_name = re.findall('./models/(.*?)\.pt', best_model_dir)
    plt.savefig(f'./results/figures/{file_name[0]}/{img_name[0]}.png')

def train_and_plot(data,dataset_name,model,best_model_dir,train_mask,val_mask,i,add_edge_weight,model_args=None):
    if model_name=='GAE' or model_name=='VGAE':
        loss_history, train_f1_history, val_f1_history,best_epoch=train_GAE(data,dataset_name, model,best_model_dir, train_mask, val_mask,add_edge_weight,model_args)
    elif model_name=='DGI' or model_name=='GRACE':
        loss_history, train_f1_history, val_f1_history,best_epoch=train_GCL(data,dataset_name,best_model_dir, train_mask, val_mask,add_edge_weight,model_args)
    else:
        loss_history, train_f1_history, val_f1_history,best_epoch=train(data,dataset_name, model,best_model_dir, train_mask, val_mask,add_edge_weight,model_args)
    plot_loss_f1(loss_history, train_f1_history,val_f1_history, best_model_dir, i)
    return best_epoch
    
def train_single_cross(data,dataset_name,train_mask,val_mask,test_mask,f1s,i,add_edge_weight,flow_iteration=0,is_flow=False,model_args=None):
    logger.info(f'cross train order: {i}')
    if is_flow:
        best_model_dir = f'./models/{model_name}_{dataset_name}_flow_iter_{flow_iteration}_cross_{i}.pt'
    else:
        best_model_dir = f'./models/{model_name}_{dataset_name}_cross_{i}.pt'

    input_dim=data.num_features
    hidden_dim=model_args[model_name][dataset_name]['hidden_dim']
    out_dim=data.num_classes
    if model_name=='GCN':
        model = GCN_NN(input_dim, hidden_dim,out_dim).to(device)
    elif model_name=='GAT':
        model = GAT_NN(input_dim, hidden_dim,out_dim,num_heads=8, num_layers=2).to(device)
        # model = GAT_NN(input_dim, hidden_dim, out_dim, num_heads=8).to(device)
    elif model_name=='GAE':
        model = GAE_NN(input_dim, hidden_dim).to(device)
        # model = GAE_NN(input_dim, hidden_dim,out_dim).to(device)
        # model = GAE(GCNEncoder(input_dim, hidden_dim,out_dim)).to(device)
    elif model_name=='DGI':
        gconv = DGI_GConv(input_dim=data.num_features, hidden_dim=hidden_dim, num_layers=2).to(device)
        encoder_model = DGI_Encoder(encoder=gconv, hidden_dim=hidden_dim).to(device)
        contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)
    elif model_name=='GRACE':
        if dataset_name=='Cora':
            activation = torch.nn.ReLU
            pe1=0.3
            pe2=0.4
            pf1=0.2
            pf2=0.4
        elif dataset_name=='CiteSeer':
            activation = torch.nn.PReLU
            pe1 = 0.3
            pe2 = 0.2
            pf1 = 0.2
            pf2 = 0.0
        elif dataset_name=='Amazon_Photo':
            activation = torch.nn.PReLU
            pe1 = 0.5
            pe2 = 0.3
            pf1 = 0.1
            pf2 = 0.1
        elif dataset_name=='WikiCS':
            activation = torch.nn.RReLU
            pe1 = 0.2
            pe2 = 0.3
            pf1 = 0.1
            pf2 = 0.1
        elif dataset_name=='Coauthor_CS':
            activation = torch.nn.RReLU
            pe1 = 0.3
            pe2 = 0.3
            pf1 = 0.2
            pf2 = 0.4
        aug1 = A.Compose([A.EdgeRemoving(pe=pe1), A.FeatureMasking(pf=pf1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=pe2), A.FeatureMasking(pf=pf2)])

        gconv = GRACE_GConv(input_dim=data.num_features, hidden_dim=hidden_dim, activation=activation, num_layers=2).to(device)
        encoder_model = GRACE_Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim=32).to(device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    if model_name == 'GAE':
        train_GAE(data, dataset_name, model,best_model_dir, train_mask, val_mask,
                                                                               add_edge_weight, model_args)
    elif model_name == 'DGI' or model_name == 'GRACE':
        train_GCL(data, dataset_name, encoder_model,contrast_model,best_model_dir,train_mask, val_mask, add_edge_weight,
                                                                               model_args)
    else:
        train(data, dataset_name, model, best_model_dir,train_mask, val_mask, add_edge_weight,
                                                                           model_args)

    if model_name=='GAE':
        # data = train_val_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
        test_f1 = test_GAE(data, model, best_model_dir, train_mask, val_mask, test_mask)
        f1s.append(test_f1)
    elif model_name=='DGI' or model_name=='GRACE':
        test_f1 = test_GCL(data,dataset_name, encoder_model,contrast_model, best_model_dir, train_mask, val_mask, test_mask)
        f1s.append(test_f1)
    else:
        test_f1 = test(data, model, best_model_dir, train_mask, val_mask, test_mask)
        f1s.append(test_f1)

def run_single_dataset(dataset_name,data,add_edge_weight,flow_iteration=0,is_flow=False,model_args=None):
    f1s = []
    cross_num=args_all['cross_nums'][args_all["choices"]["cross_num"]]
    if dataset_name == 'WikiCS':
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        for i in range(cross_num["WikiCS"]):
            train_single_cross(data, dataset_name, train_mask[:, i], val_mask[:, i], test_mask, f1s, i, add_edge_weight,
                                   flow_iteration=flow_iteration, is_flow=is_flow,model_args=model_args)
    elif dataset_name == 'Cora' or dataset_name == 'CiteSeer' or dataset_name == 'PubMed':
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        train_single_cross(data, dataset_name, train_mask, val_mask, test_mask, f1s, 0, add_edge_weight,
                               flow_iteration=flow_iteration, is_flow=is_flow,model_args=model_args)
    else:
        # train_masks, val_masks, test_masks = split_data(data)
        train_masks, val_masks, test_masks =load_split_mask(dataset_name)
        for i in range(cross_num["others"]):
            train_single_cross(data, dataset_name, train_masks[i], val_masks[i], test_masks[i], f1s, i,
                               add_edge_weight,
                               flow_iteration=flow_iteration, is_flow=is_flow,model_args=model_args)
    output_f1s(f1s)
    return f1s

def output_f1s(f1s):
    if isinstance(f1s[0], tuple):
        auc_list = [f1[0] for f1 in f1s]
        ap_list = [f1[1] for f1 in f1s]
        
        mean_value_auc = np.mean(auc_list)
        std_deviation_auc = np.std(auc_list)
        mean_value_ap = np.mean(ap_list)
        std_deviation_ap = np.std(ap_list)
        logger.info(
            f'avg_auc: {mean_value_auc}, standard deviation: {std_deviation_auc}')
        logger.info(
            f'avg_ap: {mean_value_ap}, standard deviation: {std_deviation_ap}')
        logger.info(', '.join([str(auc) for auc in auc_list]))
        logger.info(', '.join([str(ap) for ap in ap_list]))
    else:
        mean_value = np.mean(f1s)
        std_deviation = np.std(f1s)
        logger.info(f'avg: {mean_value}, standard deviation: {std_deviation}')
        logger.info(', '.join([str(f1) for f1 in f1s]))
        return mean_value,std_deviation

def run_mutiple_dataset(datasets,model_args,add_edge_weight):
    # add_edge_weight is True,  then training with weight, if  add_edge_weight is False,  then training with no weight
    from itertools import product
    for dataset_name in datasets:
        current_time=datetime.now().strftime('%Y%m%d_%H%M%S')
        change_log_file(logger, f'./logs/{current_time}_{model_name}_{dataset_name}.log')
        logger.info(f'dataset: {dataset_name}')
        print(model_args[model_name][dataset_name])
        data=load_data(dataset_name,device)
        if add_edge_weight:
            # from learner_gcn import RicciFlowTrainer
            f1s_list = []
            mean_stds=[]
            ricci_flow_args=args_all['ricci_flow_args'][args_all["choices"]["ricci_flow"]]
            iterations = ricci_flow_args['iterations']
            lrs = ricci_flow_args['lrs']
            alphas = ricci_flow_args['alphas']
            args=list(product(lrs, alphas,iterations))

            f1s = run_single_dataset(dataset_name, data, add_edge_weight, flow_iteration=0,is_flow=True,model_args=model_args)
            logger.info(f'Baseline result:')
            mean_value,std_deviation=output_f1s(f1s)
            mean_stds.append((0,0,0,mean_value,std_deviation))
            error_list=[]
            cnt=0
            for arg in args:
                lr=arg[0]
                alpha=arg[1]
                i=arg[2]
                try:
                    logger.info(f'arg order: {cnt}')
                    logger.info(f'lr: {lr}, alpha: {alpha}, iteration: {i}')
                    # load updated weight_matrix
                    weight_matrix_new=np.load(f'./data/graphs/nbr_matrix/{dataset_name}_lr{lr}_alpha{alpha}_iter{i}.npz')
                    dense_matrix=torch.from_numpy(weight_matrix_new['matrix']).float()
                    weights=dense_matrix[data.edge_index[0],data.edge_index[1]]
                    data.edge_weight=weights.to(device=device)
                    f1s=run_single_dataset(dataset_name, data,add_edge_weight,flow_iteration=i,is_flow=True,model_args=model_args)
                    logger.info(f'lr: {lr}, alpha: {alpha}, iteration: {i}, result:')
                    mean_value,std_deviation=output_f1s(f1s)
                    f1s_list.append((lr,alpha,i,f1s))
                    mean_stds.append((lr,alpha,i,mean_value,std_deviation))
                except:
                    logger.info(f'error arg: lr: {lr}, alpha: {alpha}, iteration: {i} failed')
                    error_list.append((lr,alpha,i))
                    max_avg=0
                    std=0
                    for mean_std in mean_stds:
                        if mean_std[3]>max_avg:
                            max_avg=mean_std[3]
                            std=mean_std[4]
                    logger.info(f'best result: avg: {max_avg}, std: {std}')
                    return max_avg,std
                cnt+=1
            logger.info(f'overall result:')
            max_avg=0
            best_res=None
            for f1s in f1s_list:
                logger.info(f'lr: {f1s[0]}, alpha: {f1s[1]}, iter:{f1s[2]}, result:')
                mean_value,std_deviation=output_f1s(f1s[3])
                if mean_value>max_avg:
                    max_avg=mean_value
                    best_res=f1s
            logger.info(f'best result:')
            logger.info(f'lr: {best_res[0]}, alpha: {best_res[1]}, iter:{best_res[2]}, result:')
            mean_value,std_deviation=output_f1s(best_res[3])
            for error in error_list:
                logger.info(f'lr: {error[0]}, alpha: {error[1]}, iter:{error[2]} failed')
            return mean_value,std_deviation
        else:
            f1s=run_single_dataset(dataset_name, data,add_edge_weight,model_args=model_args)
            mean_value,std_deviation=output_f1s(f1s)
            return mean_value,std_deviation
def main():
    start_time = time.time()
    logger.info(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    datasets = [args_all["dataset_names"][args_all["choices"]["dataset_name"]]]
    add_edge_weight = args_all["choices"]["add_edge_weight"]
    model_args_debug = args_all['model_args_debugs'][args_all["choices"]["model_arg_debug"]]
    hidden_dim = model_args_debug['hidden_dim']
    learning_rate = model_args_debug['learning_rate']
    weight_decay = model_args_debug['weight_decay']
    epochs = model_args_debug['epochs']
    round=args_all["choices"]["round"]

    from itertools import product
    args = list(product(hidden_dim, learning_rate, weight_decay,epochs))
    logger.info(f'dataset: {datasets[0]}, add_edge_weight: {add_edge_weight}, round: {round}')
    model_args = args_all['model_args']

    for arg in args:
        logger.info(f'hidden_dim: {arg[0]}, learning_rate: {arg[1]}, weight_decay: {arg[2]}, epochs: {arg[3]}')
        model_args[model_name][datasets[0]]['hidden_dim'] = arg[0]
        model_args[model_name][datasets[0]]['learning_rate'] = arg[1]
        model_args[model_name][datasets[0]]['weight_decay'] = arg[2]
        model_args[model_name][datasets[0]]['epochs'] = arg[3]
        mean_values_rounds = []
        for i in range(round):
            logger.info(f'round: {i}')
            try:
                mean_value,std_deviation=run_mutiple_dataset(datasets,model_args,add_edge_weight)
                # mean_values.append(mean_value)
                mean_values_rounds.append(mean_value)
            except:
                import traceback
                traceback.print_exc()
                continue
        mean_mean_value_rounds = np.mean(mean_values_rounds)
        mean_std_deviation_rounds = np.std(mean_values_rounds)
        logger.info(f'hidden_dim: {arg[0]}, learning_rate: {arg[1]}, weight_decay: {arg[2]}, epochs: {arg[3]}')
        logger.info(f'mean_mean_value_rounds: avg: {mean_mean_value_rounds}, std: {mean_std_deviation_rounds}')
        print(mean_values_rounds)
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Ending time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logger.info(f"Run time: {runtime} seconds")

if __name__ == '__main__':
    main()
