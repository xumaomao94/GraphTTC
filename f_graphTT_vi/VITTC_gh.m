function [A_completed,Gcore,Lambda,Tau,rse,rank_estimated,Power] = VITTC_gh(A_raw,A_observed,Mask,Lap,varargin)
% ------------------------------------------------------
% Graph Regularized Tensor Train Completion, modeled by graph+GH-prior,
% estimate the posteriors under the mean-field approximation. This code
% realize the VI procedure introduced in the following reference.
% 
% Reference
%
% ------------------Input------------------
% A_raw
%       The original tensor data, only used for performance test, use
%       A_observed instead if there is no A_raw
% A_observed
%       The noisy, incomplete tensor data
% Mask
%       Indicating tensor,
%       -1 -> entry observed
%       -0 -> not observed
% Lap
%       The graph Laplacian, with number of cells same as the order of
%       A_observed. Lap{d} is the graph Laplacian on the d-th mode of the
%       tensor. Set Lap{d} as an identity matrix if there is no prior
%       information on the d-th mode.
% maxiter (default: 100)
%       Max number of iterations for the VI update
% initmethod (default: 'ttsvd')
%       -'ttsvd'-> use Gaussian random variables to fill in empty entries
%       -'randomize'-> all entries are initialized by Gaussian variables
% maxrank (default: 64)
%       max rank for the TT ranks
% thre_rank_prune (default: 1e-5)
%       Slices with (average power/average core power) less than thre_rank_prune will be discarded
%       Recommended value: smaller than 1e-2, e.g., 1e-3, 1e-5
% thre_stop (default: 1e-7)
%       stop the iteration when relative square error between current recovered tensor and
%       last update is smaller than thre_stop
% show_info (default: false)
%       -true -> print information during the algorithm
%       -false -> not to print
% ------------------Output------------------
% A_completed
%       Estimated tensor data
% Gcore
%       Estimated TT cores
% Lambda
%       The mean of the GIG varaibles
% Tau
%       The estimated noise power
% rse
%       rse between A_raw & A_completed
% rank_estimated
%       The estimated rank
% Power
%       The average power of TT core slices w.r.t. each TT rank
% 
% ------------------------------------------------------
% XU Le, 10th Sep 2022, Last Update: 10th Sep 2022
% ------------------------------------------------------
%%  read/set parameters
p = inputParser;
defaultPar.Maxiter = 100;
defaultPar.InitialMethod = 'ttsvd';
defaultPar.MaxRank = 64;
defaultPar.RankPruneThre = 1e-5;
defaultPar.IterEndThre = 1e-7;
defaultPar.ShowInfo = false;

addRequired(p,'A_raw',@isnumeric);
addRequired(p,'A_observed',@isnumeric);
addRequired(p,'Mask',@(x) (isnumeric(x)||islogical(x)) && isequal(size(x),size(A_observed)));
addRequired(p,'Lap',@(x) iscell(x) && length(x)==ndims(A_observed));
addOptional(p,'maxiter',defaultPar.Maxiter,@isscalar);
addOptional(p,'initmethod',defaultPar.InitialMethod,@(x) ismember(x,{'ttsvd','randomize'}));
addOptional(p,'maxrank',defaultPar.MaxRank,@isscalar);
addOptional(p,'thre_rankprune',defaultPar.RankPruneThre,@isscalar);
addOptional(p,'thre_stop',defaultPar.IterEndThre,@isscalar);
addOptional(p,'show_info',defaultPar.ShowInfo,@islogical);
parse(p,A_raw,A_observed,Mask,Lap,varargin{:});

Par = p.Results;
if Par.show_info
    tic
    fprintf('-------------- GraphTT_VI begins --------------\n')
    disp(Par)
end

%% initialization
Size_A = size(A_observed);
indnorm = 10^(ndims(A_observed)-1)/max( abs(reshape(A_observed,[],1)) );
A_observed = A_observed.*indnorm;
rse = zeros(1,Par.maxiter+1);
[Lambda,Tau,Gcore] = VITTC_GH_init_graph_ind(A_observed,Mask,Lap,Par.maxrank,Par.initmethod);

%% VI update
A_Ctemp = 0;
for i = 1:Par.maxiter
    if Par.show_info
        fprintf('Iteration: %d; Time: %f; ',i,toc)
    end
    
    [Gcore,V_final,W_final] = update_gcore_GH_graph_ind(A_observed,Mask,Gcore,Lambda,Tau,Lap);
    Lambda = update_lambda_GH_graph_ind(A_observed,Gcore,Lambda,Tau,Lap);
    Tau = update_tau_GH_ind(A_observed,Mask,Gcore,Lambda,Tau,V_final,W_final);
    if Par.thre_rankprune >= 0.1 && i == 1
        fprintf('the threshold for rank pruning <thre_rankprune> is not reasonable, better set as values smaller than 1e-2\n')
    end
    [Gcore,Lambda] = rank_reduce_relative_GH_graph_ind_Power(ndims(A_observed),Gcore,Lambda,Par.thre_rankprune);

    A_completed = tt2full(Gcore,Size_A)./indnorm;
    rse(i) = sumsqr(A_completed-A_raw)/sumsqr(A_raw);
    dist = sumsqr(A_completed(:)-A_Ctemp(:))/sumsqr(A_completed(:));
    if Par.show_info
        fprintf('rse between current and last update: %.9f\n',dist)
    end
    if dist < Par.thre_stop
        if Par.show_info
            fprintf('-------------- GraphTT_VI converges --------------\n')
        end
        break
    end
    A_Ctemp = A_completed;
end

%% Evaluate
A_completed = tt2full(Gcore,Size_A);
A_completed = A_completed./indnorm;
rse(end) = sumsqr(A_completed-A_raw)/sumsqr(A_raw);

ndims_A  = ndims(A_observed);
Power_L = cell(1,ndims_A);
Power_H = cell(1,ndims_A);
for order = 1:ndims_A
    Power_L{order} = zeros(size(Lambda.mean{order}));
    Power_H{order} = zeros(size(Lambda.mean{order+1}));
    meansqr = Gcore.mean{order}(:)'*Gcore.mean{order}(:)/numel(Gcore.mean{order});
    for r = 1:length(Power_L{order})
        gcoreslice = Gcore.mean{order}(r,:,:);
        Power_L{order}(r) = gcoreslice(:)'*gcoreslice(:)/meansqr/numel(gcoreslice);
    end
    for r = 1:length(Power_H{order})
        gcoreslice = Gcore.mean{order}(:,r,:);
        Power_H{order}(r) = gcoreslice(:)'*gcoreslice(:)/meansqr/numel(gcoreslice);
    end
end
Power = cell(1,ndims_A); Power{1} = 1;
for order = 2:ndims_A
    Power{order} = Power_L{order}+Power_H{order-1};
    if Par.show_info
%         subplot(1,ndims_A-1,order-1); bar(Power{order});
    end
end

%% the final guessed rank
rank_estimated = ones(ndims(A_observed)+1,1);
for order = 1:ndims(A_observed)
    rank_estimated(order) = size(Gcore.mean{order},1);
end

end