function [A_completed,Gcore,rse] = ttc_graph(A_raw,A_observed,Mask,Lap,beta,rank_init,varargin)
% ------------------------------------------------------
% Graph Regularized Tensor Train Completion, solved under the BCD framework.
% This code realizes the optimization method introduced in the following reference.
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
% beta
%       The regularization parameter(s)
%       -input only one number
%           Recognized as beta_0 as introduced in the ref paper. beta_d's
%           will be automatically set by (beta/power of d-th TT core)
%       -input a vector with length = dimension of the observed tensor
%           beta(d) will be the regularization parameter for the d-th mode
% rank_init
%       The initialized TT ranks, must follow the structure
%       [1,r2,...,rD,1], which is with length (1+dimension of the observed tensor)
% maxiter (default: 100)
%       Max number of iterations for the als update
% initmethod (default: 'ttsvd')
%       -'ttsvd'-> use Gaussian random variables to fill in empty entries
%       -'randomize'-> all entries are initialized by Gaussian variables
% thre_stop (default: 1e-6)
%       stop the iteration when relative square error between current recovered tensor and
%       last update is smaller than thre_stop
% update_method (default: 'fiber_als')
%       - 'fiber_als' -> update TT core fibers as the basic block, as
%       introduced in the reference paper
%       - 'core_als' -> update TT cores as the basic block, for comparison
% show_info (default: false)
%       -true -> print information during the algorithm
%       -false -> not to print
% ------------------Output------------------
% A_completed
%       Estimated tensor data
% Gcore
%       Estimated TT cores
% rse
%       rse between A_raw & A_completed
% 
% ------------------------------------------------------
% XU Le, 10th Sep 2022, Last Update: 10th Sep 2022
% ------------------------------------------------------
%%  read/set parameters
p = inputParser;
defaultPar.Maxiter = 100;
defaultPar.InitialMethod = 'ttsvd';
defaultPar.IterEndThre = 1e-6;
defaultPar.UpdateMethod = 'fiber_als';
defaultPar.ShowInfo = false;


addRequired(p,'A_raw',@isnumeric);
addRequired(p,'A_observed',@isnumeric);
addRequired(p,'Mask',@(x) (isnumeric(x)||islogical(x)) && isequal(size(x),size(A_observed)));
addRequired(p,'Lap',@(x) iscell(x) && length(x)==ndims(A_observed));
addRequired(p,'beta',@(x) isnumeric(x) && (isscalar(x)||length(x)==ndims(A_observed)));
addRequired(p,'rank_init',@(x) isnumeric(x) && length(x)==1+ndims(A_observed));

addOptional(p,'maxiter',defaultPar.Maxiter,@isscalar);
addOptional(p,'initmethod',defaultPar.InitialMethod,@(x) ismember(x,{'ttsvd','randomize'}));
addOptional(p,'thre_stop',defaultPar.IterEndThre,@isscalar);
addOptional(p,'update_method',defaultPar.UpdateMethod,@(x) ismember(x,{'fiber_als','core_als'}));
addOptional(p,'show_info',defaultPar.ShowInfo,@islogical);

parse(p,A_raw,A_observed,Mask,Lap,beta,rank_init,varargin{:});

Par = p.Results;
if Par.show_info
    tic
    fprintf('-------------- GraphTT_opt begins --------------\n')
    disp(Par)
end

%% Initialization
Size_A = size(A_observed);
ndims_A = ndims(A_observed);
indnorm = 1/max(A_observed(:));
A_observed = A_observed.*indnorm;
rse = zeros(1,Par.maxiter+1);

[Gcore,X] = ttc_graph_init(A_observed,Mask,rank_init,Par.initmethod);
R = zeros(1,ndims_A+1); % reload the ranks, in case the initial ranks are set larger than required
R(1) = 1; R(end) = 1;
for order = 2:ndims_A
    R(order) = size(Gcore{order-1},2);
end

if isscalar(beta) % set beta_0 only, get the beta_d according to that introduced in the reference paper
    weight = zeros(1,ndims_A);
    for order = 1:ndims(A_observed)
        weight(order) = sum(Gcore{order}(:).^2)/R(order)/R(order+1);
    end
    beta_init = beta./weight;
else
    beta_init = beta; % manually set all beta_d's
end

%% GraphTT-opt algorithm
A_Ctemp = 0;
for i = 1:Par.maxiter
    if Par.show_info
        if mod(i,20) == 1
            fprintf('The %d-th iteration; time: %f; ',i,toc)
        end
    end
    
    if strcmp(Par.update_method,'fiber_als')
        [Gcore,X] = als_fiber(A_observed,Mask,Gcore,Lap,beta_init,X);
    elseif strcmp(Par.update_method,'core_als')
        [Gcore,X] = als_core(A_observed,Mask,Gcore,Lap,beta_init,X);
    end
    
    
    A_completed = tt2full_4ttc(Gcore,Size_A)./indnorm;
    
    rse(i) = sumsqr(A_completed-A_raw)/sumsqr(A_raw);
%     objfun = sumsqr(S.*(A_completed-A_raw));
%     for order = 1:ndims(A_observed)
%         Gcore2mat = reshape(Gcore{order},[R(order)*R(order+1),Size_A(order)]);
%         for r = 1:R(order)*R(order+1)
%             objfun = objfun + beta_init(order) * Gcore2mat(r,:) * Lap{order} * Gcore2mat(r,:)';
%         end
%     end

    dist = sumsqr(A_completed-A_Ctemp)/sumsqr(A_completed);
    if Par.show_info
        if mod(i,20) == 1
            fprintf('rse between the current and last update: %.9f\n', dist)
        end
    end
    if dist < Par.thre_stop
        if Par.show_info
            fprintf('-------------- GraphTT_opt converges --------------\n')
        end
        break
    end
    A_Ctemp = A_completed;
end

%% Evaluate
A_completed = tt2full_4ttc(Gcore,Size_A)./indnorm;
rse(end) = sumsqr(A_completed-A_raw)/sumsqr(A_raw);
end