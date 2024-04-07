clear
dataset_name='Coil20';
dlmwrite('./result/'+ "" +dataset_name+ "" +'_after_update.txt',['start cluster:'],'-append','delimiter','','newline','pc');
 % load('./data/YALE_165n_1024d_15c_uni.mat');
 load('./data/'+""+dataset_name+ "" +'.mat');
 X=X';
[m,n]=size(X);
k=10;
%mu = .5; 
%alpha = 0.01;
%beta = 0.01;
alphalist=1/sqrt(max([m n]));
args=load('./args/'+ ""+dataset_name+ ""+'.txt');
args_num=length(args);

lrs=[0.5,0.6,0.7,0.8,0.9,1.0];
iterations=[1,2,3,4,5,6,7];
alphas_rc=[0.1,0.2,0.3,0.4,0.5];

c = length(unique(y));

for ii = 1:length(alphalist)
    alpha = alphalist(ii);
    for jj = 1:args_num
        beta = args(jj);
        mu= args(args_num+jj);
        for lr_cnt=1:length(lrs)
           for alpha_rc_cnt=1:length(alphas_rc)
                for iteration_cnt=1:length(iterations)
                    lr=lrs(lr_cnt);
                    alpha_rc=alphas_rc(alpha_rc_cnt);
                    iteration=iterations(iteration_cnt);
                    S_filename='./S_update/'+"" +dataset_name+ "" +'_b'+""+beta+""+'_m'+""+mu+""+'_lr'+""+lr+""+"_alpha"+alpha_rc+""+'_iter'+""+iteration+""+'.mat';
                    disp(S_filename)
                    if exist(S_filename, 'file') ~= 2
                        continue;
                    end
                    disp([alpha,beta,mu,lr,alpha_rc,iteration])
                    load(S_filename);
                    actual_ids = spectral_clustering(S, c);
                    result=ClusteringMeasure(actual_ids ,y)
                
                    dlmwrite('./result/'+ "" +dataset_name+ "" +'_after_update.txt',[alpha,beta,mu,lr,alpha_rc,iteration,result],'-append','delimiter','\t','newline','pc');

                end
            end
        end
    end
end