clear
dataset_name='YALE';
dlmwrite('./result/'+ "" +dataset_name+ "" +'.txt',['start training:'],'-append','delimiter','','newline','pc');
 % load('./data/YALE_165n_1024d_15c_uni.mat');
 load('./data/'+""+dataset_name+ "" +'.mat');
 X=X';
[m,n]=size(X);
k=10;
%mu = .5; 
%alpha = 0.01;
%beta = 0.01;
alphalist=1/sqrt(max([m n]));
%   alphalist=[1 20 ];
% alphalist=[1e-12, 1e-11,1e-10, 1e-9, 1e-8, 1e-7,1e-6,1e-5,1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100];
% alphalist=[1e-6,1e-5,1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100];
% betalist = [0.001, 0.01,0.1, 1, 5, 10, 15, 20];
% mulist = [0.001, 0.01,0.1, 1, 5, 10, 15, 20];

betalist = [1e-7,1e-6,1e-5,1e-4,1e-3];
mulist = [1,20];

% betalist = [1,10,20,30,40,50,60,70,80,90,100];
% mulist = [1,10,20,50,100,200,1000,2000,10000];

% Y1 =zeros(m,n) ;
% Y2 =zeros(m,n) ;
% E = zeros(m,n);
% Z = X;
% c = length(unique(y));
% distX = L2_distance_1(X,X);
% [distX1, idx] = sort(distX,2);


for ii = 1:length(alphalist)
    alpha = alphalist(ii);
    for jj = 1:length(betalist)
        beta = betalist(jj);
        % [gamma] = cal_gamma(X,distX1,beta,k);
         for ij = 1:length(mulist)
            mu= mulist(ij);
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
        
            actual_ids = spectral_clustering(S, c);
        
            result=ClusteringMeasure(actual_ids ,y)
    
            % dlmwrite('./result/yale.txt',[alpha,beta,mulist(ij),result],'-append','delimiter','\t','newline','pc');
           % dlmwrite('./result/yale2.txt',[alpha,beta,mulist(ij),result],'-append','delimiter','\t','newline','pc');
            % dlmwrite('./result/coil20.txt',[alpha,beta,mulist(ij),result],'-append','delimiter','\t','newline','pc');
            % dlmwrite('./result/ORL.txt',[alpha,beta,mulist(ij),result],'-append','delimiter','\t','newline','pc');
           dlmwrite('./result/'+ "" +dataset_name+ "" +'.txt',[alpha,beta,mulist(ij),result],'-append','delimiter','\t','newline','pc');
           % distX=distX_orgional
           %  distX1=distX1_original
           %  idx=idx_original
      end
end
end


    

    
    
    
