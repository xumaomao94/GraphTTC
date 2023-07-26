function [Gcore,X] = ttc_graph_init(A_observed,Mask,rank_init,initmethod)
    
    Size_A = size(A_observed);
    ndims_A = ndims(A_observed);
    
    R_init = rank_init;
    
    Mask_index = find(Mask~=0);
    
    if strcmp(initmethod,'randomize')
        A_guess = normrnd(mean(A_observed(Mask_index)),std(A_observed(Mask_index)),Size_A);% normal random, with A_mean and A_var
    elseif strcmp(initmethod,'ttsvd')
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
        
%         % update W
%         i_locate = order+1;
%         i_before = i_locate -1;
%         W{i_locate} = W{i_before} * reshape(Gcore{i_before},[R(i_before),R(i_before+1)*Size_A(i_before)]);
%         W{i_locate} = reshape(W{i_locate},[prod(Size_A(1:i_before-1)),R(i_before+1),Size_A(i_before)]);
%         W{i_locate} = permute(W{i_locate},[1,3,2]);
%         W{i_locate} = reshape(W{i_locate},[prod(Size_A(1:i_before)),R(i_before+1)]);
    end
end