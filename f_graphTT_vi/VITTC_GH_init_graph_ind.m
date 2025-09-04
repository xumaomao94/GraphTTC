function [lambda,tau,Gcore,E,T] = VITTC_GH_init_graph_ind(A_observed,Mask,Lap,maxrank,initmethod,isOutlier)
    Size_A = size(A_observed);
    ndims_A = ndims(A_observed);

    % Gcore mean, determine R at the same time
    R = zeros(ndims_A+1,1);
    R(1) = 1; R(end) = 1;
    Gcore.mean = cell(1,ndims_A);
    Gcore.var = cell(1,ndims_A);
    
    
    Mask_index = find(Mask~=0);
    if strcmp(initmethod,'randomize')
        A_guess = normrnd(mean(A_observed(Mask_index)),std(A_observed(Mask_index)),Size_A);% normal random, with A_mean and A_var
    elseif strcmp(initmethod,'ttsvd')
        A_guess = A_observed.*Mask+(1-Mask).*normrnd(mean(A_observed(Mask_index)),std(A_observed(Mask_index)),Size_A);
    end

    Aremain_M = reshape(A_guess,Size_A(1),[]); % use A_observed + random guess as initialization
    for i = 1:ndims_A-1
        [U,S,V] = svd(Aremain_M,'econ');
        R(i+1) = min(size(S,1),maxrank);
        U = U*S.^(1/(ndims_A-i+1));
        S = S.^((ndims_A-i)/(ndims_A-i+1));
        Gcore.mean{i} = permute(reshape(U(:,1:R(i+1)),[R(i),Size_A(i),R(i+1)]),[1,3,2]);
        Aremain_M = reshape(S(1:R(i+1),1:R(i+1))*V(:,1:R(i+1))',[R(i+1)*Size_A(i+1),prod(Size_A(i+1:end))/Size_A(i+1)]);
    end
    Gcore.mean{ndims_A} = reshape(Aremain_M,[R(ndims_A),R(ndims_A+1),Size_A(ndims_A)]);
    
    
    smallvalue = 10^(-6);
    % lambda
    lambda.b = smallvalue; %0;%10^(-9); %
    lambda.c = smallvalue;%-1; % %-10^(-6); -min(Size_A)
    
    lambda.a = cell(1,ndims_A+1);
    lambda.a(1) = {1}; lambda.a(ndims_A+1) = {1};
    lambda.a_alpha = cell(1,ndims_A+1); lambda.a_alpha(1) = {1}; lambda.a_alpha(ndims_A+1) = {1};
    lambda.a_beta = cell(1,ndims_A+1); lambda.a_beta(1) = {1}; lambda.a_beta(ndims_A+1) = {1};
    for i = 2:ndims_A
        lambda.a_alpha(i) = {(smallvalue-min(0,lambda.c/2))*ones(1,R(i))};
        lambda.a_beta(i) = {smallvalue*ones(1,R(i))};
        lambda.a(i) = {lambda.a_alpha{i}./lambda.a_beta{i}};
    end
    lambda.meaninv = cell(1,ndims_A+1); % for conviency, we set lambda(0) = 1 and lambda(N) = 1. The real lambda(i) is denoted as lambda(i+1) here!
    lambda.meaninv(1) = {1}; lambda.meaninv(end) = {1};
    lambda.mean = cell(1,ndims_A+1);
    lambda.mean(1) = {1}; lambda.mean(end) = {1};
    
    for i = 2:ndims_A
        lambda.mean(i) = {sqrt(lambda.b)./sqrt(lambda.a{i}) .* divbessel(lambda.c,sqrt(lambda.b.*lambda.a{i}))};
        lambda.meaninv(i) = {sqrt(lambda.a{i})./sqrt(lambda.b) .* divbessel(lambda.c,sqrt(lambda.b.*lambda.a{i})) - 2*lambda.c/lambda.b};

        lambda.mean{i} = ones(1,R(i));
        lambda.meaninv{i} = ones(1,R(i));
    end
    
    % tau
    tau.a = smallvalue;
    tau.b = smallvalue;
    tau.mean = tau.a/tau.b;
    tau.var = tau.a/(tau.b^2);
    
    for i = 1:ndims_A
        for p = 1:R(i)
            for q = 1:R(i+1)
                Gcore.var{i}(p,q,:,:) = 1./lambda.meaninv{i}(p)*lambda.meaninv{i+1}(q)*Lap{i};
            end
        end
    end

    % E and T
    if ~isOutlier
        E.mean = 0;
        T.mean = 0;
    else
        T.a = smallvalue * ones(Size_A);
        T.b = smallvalue * ones(Size_A);
        T.mean = T.a ./ T.b;
        E.mean = zeros(Size_A);
        E.var = 1./T.mean;
    end

end