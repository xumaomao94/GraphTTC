function [Gcore,X,beta_init,beta_equ] = ttc_graph_init(A_observed,Mask,rank_init,beta,indnorm,Lap,initmethod,thre_stop)
    
    Size_A = size(A_observed);
    ndims_A = ndims(A_observed);
    
    R_init = rank_init;
    
    Mask_index = find(Mask~=0);
    %% init Gcore
    if strcmp(initmethod,'fromVI')
        A_temp = A_observed./indnorm;
        [~,~,Lambda,Tau,~,~,~] = VITTC_gh(A_temp,A_temp,Mask,Lap,...
                        'maxiter',200,...
                        'maxrank',max(rank_init),...%                         'thre_rankprune',0,... % not prune ranks
                        'thre_stop',thre_stop,...
                        'show_info',false);
        for i = 2:ndims_A
            R_init(i) = length(Lambda.meaninv{i});
        end
    end
    
    if strcmp(initmethod,'randomize')
        A_guess = normrnd(mean(A_observed(Mask_index)),std(A_observed(Mask_index)),Size_A);% normal random, with A_mean and A_var
    elseif strcmp(initmethod,'ttsvd') || strcmp(initmethod,'fromVI')
        A_guess = A_observed.*Mask+(1-Mask).*normrnd(mean(A_observed(Mask_index)),std(A_observed(Mask_index)),Size_A);
    end

    Aremain_M = reshape(A_guess,Size_A(1),[]); % use A_observed + random guess as initialization
    Gcore = cell(1,ndims_A);
    R = zeros(ndims_A+1,1);
    R(1) = 1; R(end) = 1;
    for i = 1:ndims_A-1
        [U,S,V] = svd(Aremain_M,'econ');
        R(i+1) = min(size(S,1),R_init(i+1));
        U = U*S.^(1/(ndims_A-i+1));
        S = S.^((ndims_A-i)/(ndims_A-i+1));
        Gcore{i} = permute(reshape(U(:,1:R(i+1)),[R(i),Size_A(i),R(i+1)]),[1,3,2]);
        Aremain_M = reshape(S(1:R(i+1),1:R(i+1))*V(:,1:R(i+1))',[R(i+1)*Size_A(i+1),prod(Size_A(i+1:end))/Size_A(i+1)]);
    end
    Gcore{ndims_A} = reshape(Aremain_M,[R(ndims_A),R(ndims_A+1),Size_A(ndims_A)]);
    
    X = cell(1,ndims_A); % to store the mean's product after the current core; [R_{d+1}, J_{d+1}...J_D]
    X{ndims_A} = 1;
    W = cell(1,ndims_A); % to store the mean's product before the current core; [J1...J_{d-1},R_{d}]
    W{1} = 1;

    for order = 1:ndims_A-1
        i_locate = ndims_A-order; % from ndims_A-1 to 1
        i_after = i_locate+1; % index of the core after i_locate

        % update X
        X{i_locate} = reshape(permute(Gcore{i_after},[1,3,2]),[R(i_after)*Size_A(i_after),R(i_after+1)]) * X{i_after};
        X{i_locate} = reshape(X{i_locate},[R(i_after),prod(Size_A(i_after:end))]);
    end
    
    %% init beta_init
    weight = zeros(1,ndims_A);
    for order = 1:ndims(A_observed)
        weight(order) = sum(Gcore{order}(:).^2)/R(order)/R(order+1);
    end
            
    if strcmp(initmethod,'fromVI') % this one is only for the test of reuse of beta obtained by graphTT-vi
        beta_init = zeros(1,ndims_A);
        for i = 1:length(beta_init)
            beta_init(i) = majorvote(Lambda.meaninv{i}) ...
                * majorvote(Lambda.meaninv{i+1}) / Tau.mean ...
                * Tau.indnorm^(2/ndims_A - 2);
        end
        beta_init = beta_init ./ indnorm^(2/ndims_A-2);
        beta_init = mean(beta_init).*ones(size(beta_init));
        beta_equ = beta_init.*weight;
    else
        if isscalar(beta) % set beta_0 only, get the beta_d according to that introduced in the reference paper
            beta_init = beta./weight;
            beta_equ = beta * ones(size(beta_init));
        else
            beta_init = beta; % manually set all beta_d's
            beta_equ = beta.*weight;
        end
    end
    
    
end