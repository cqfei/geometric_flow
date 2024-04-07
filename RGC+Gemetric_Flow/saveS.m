clear
dataset_name='BA';
% dlmwrite('./result/'+ "" +dataset_name+ "" +'.txt',['start training:'],'-append','delimiter','','newline','pc');
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

for ii = 1:length(alphalist)
    alpha = alphalist(ii);
    for jj = 1:args_num
        beta = args(jj);
        mu= args(args_num+jj);
        save_file_name='./S_mat/'+""+dataset_name+""+"_b"+beta+"_m"+mu+".mat";
        disp([alpha,beta,mu])

        Y1 =zeros(m,n) ;
        Y2 =zeros(m,n) ;
        E = zeros(m,n);
        Z = X;
        c = length(unique(y));
        distX = L2_distance_1(X,X);
        [distX1, idx] = sort(distX,2);
        [gamma] = cal_gamma(X,distX1,beta,k);

        for i = 1:200
            D =  updateD(E,X,Y1,Y2,mu,Z,gamma);
            distX = L2_distance_1(D,D);
            [distX1, idx] = sort(distX,2);
            [gamma] = cal_gamma(D,distX1,beta,k);
            E = updateE(D,E,X,Y1,mu,alpha);
            S = updateS(X,distX1,idx,k,gamma,beta);
            S=(S+S')/2;
            L = diag(sum(S))-S;
            Z = updateZ(L,beta,mu,D,Y2);
            Y1 = Y1+mu*(D+E-X);
            Y2 = Y2+mu*(D-Z);
            mu=mu*1.1;
        end

        save(save_file_name,"S")
    end
end


    

    
    
    