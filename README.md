

This code is the experimental code of paper "Unsupervised Dynamic Discrete Structure Learning:A Geometric Evolution Method"

## Environment Setting
Python=3.9, torch=1.12

## Instuctions
Whether in latent graph learning task or graph neural network task, you need to update the weight matrix using the model.py file.

You should prepare the dataset for graph neural network task by yourself.

## Manifold Learning task
1. learner_manifold_learning.py is used to train in manifold learning task. For example, train on dataset "gender".
   
    dataset='gender'
   
    t = Trainer()
   
    t.train_single_dataset(dataset)
   
## Latent graph learning task
Operation In RGC+Geometric_Flow/ fold. All the dataset used for latent graph learning task can be found in data/lantent_graph_learning/.

we add a geometric flow module in RGC algorithm to update S matrix. 

In our geometric flow setting, you should save the original S matrix using "saveS.m", then use "model.py" to update  the original S matrix.
Finally, you can run new clustering experiments in "update_and_cluster.m" by loading updated S matrix.

The data file in "data/latent_graph_learning/mat/".

In S_mat/, 

b=0.001,m=0.5 is the best hyperparameter that original RGC algorithm achieve the best performance in YALE dataset. We save the S matrix in this setting.
In S_update fold, we provide all update matrix for matrix YALE_b0.001_m5.mat in the flowing setting: 
lrs=[0.5,0.6,0.7,0.8,0.9,1.0];
iterations=[1,2,3,4,5,6,7];
alphas_rc=[0.1,0.2,0.3,0.4,0.5];

## Graph Neural network task
Operation In GNN+Geometric_Flow/ fold.

You should configure "choices" item in file './config.json'. Take the following configuration as example.

model_name means the training model, for example 0 represents GCN.
model_arg_debug means the order value of hyperparameter setting for hidden_dim, learning_rate, weight_decay, epochs. 
cross_num means the number of cross-validation times for different datasets.
ricci_flow means whether use gemetric flow, 0 mean True.
dataset_name means the training dataset.
device means GPU device.
add_edge_weight means whether use weight in training.
round means training round.
epoch_step means how many epochs the model is evaluated and how many epochs the best model is saved.

"choices": {
    "model_name": 0,
    "cross_num": 0,
    "model_arg_debug": 33,
    "ricci_flow": 0,
    "dataset_name": 0,
    "device": 4,
    "add_edge_weight": true,
    "round": 10,
    "epoch_step": 10
  },

A good suggestion is that you'd better update weight matrix first using "model.py" in root fold, before you use gemetric flow to update weight matrix during your training.
