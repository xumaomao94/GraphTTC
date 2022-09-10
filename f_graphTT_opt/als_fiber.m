function [Gcore,X] = als_fiber(A_observed,Mask,Gcore,Lap,beta,X)

size_A = size(A_observed);
ndims_A = ndims(A_observed);
R = zeros(1,ndims_A+1); R(1) = 1; R(end) = 1;
for i = 2:ndims_A
    R(i) = size(Gcore{i},1);
end

% the update for the order-th TTcore
W = cell(1,ndims_A); W{1} = 1; % to store the mean's product before the current core, actually only one W is needed, instead of W{1}-W{ndims_A}
    
for order = 1:ndims_A
    % update the precision matrix
%     fprintf('order: %d',order)

    if order ~= 1
        W{order} = W{order-1} * reshape(Gcore{order-1},R(order-1),R(order)*size_A(order-1));
        W{order} = reshape(permute(reshape(W{order},[prod(size_A(1:order-2)),R(order),size_A(order-1)]),[1,3,2]),[prod(size_A(1:order-1)),R(order)]);
    end

    M_2matrix = reshape(Mask,[prod(size_A(1:order-1)),size_A(order),prod(size_A(order+1:end))]);
    M_2matrix = permute(M_2matrix,[1,3,2]); % [I1...I(d-1) , I(d+1)...I(D) , I(d)]
    A_2matrix = reshape(A_observed,[prod(size_A(1:order-1)),size_A(order),prod(size_A(order+1:end))]);
    A_2matrix = permute(A_2matrix,[1,3,2]); % [I1...I(d-1) , I(d+1)...I(D) , I(d)]
%     AM_2matrix = M_2matrix.*A_2matrix;
    if ~iscell(beta)
        Gcoreprecision_Lambdapart = beta(order)*Lap{order};
    end
    Gcoreprecision_VUpart = zeros(R(order)^2,R(order+1)^2,size_A(order));
    Gcore_mean_times_precision = zeros(R(order),R(order+1),size_A(order));

    V_order = khatrirao(W{order}',W{order}'); % R^2 * I_{I_1...I_{order-1}}
    U_order = khatrirao(X{order},X{order}); % R^2 * prod(I_{order+1}...I_{D})

    for d = 1:size_A(order)
        Gcoreprecision_VUpart(:,:,d) = V_order * M_2matrix(:,:,d) * U_order';
        Gcore_mean_times_precision(:,:,d) = W{order}' * (A_2matrix(:,:,d).*M_2matrix(:,:,d)) * X{order}';
    end

    for p = 1:R(order)
        for q = 1:R(order+1)
            Gcoreprecision_pq = diag(squeeze(Gcoreprecision_VUpart((p-1)*R(order)+p,(q-1)*R(order+1)+q,:)));
            if iscell(beta) % put a matrix in
                Gcoreprecision_Lambdapart = beta{order}(p,q) * Lap{order};
            end
            Gcoreprecision_pq = Gcoreprecision_Lambdapart+Gcoreprecision_pq;
            Gcore{order}(p,q,:) = (Gcoreprecision_pq) \ squeeze(  - ( sum(sum(Gcoreprecision_VUpart((p-1)*R(order)+1:p*R(order),(q-1)*R(order+1)+1:q*R(order+1),:).*Gcore{order},1),2) - Gcoreprecision_VUpart((p-1)*R(order)+p,(q-1)*R(order+1)+q,:).*Gcore{order}(p,q,:)) ...
                + Gcore_mean_times_precision(p,q,:)  ) ;
        end
    end
end

for order_inv = 2:ndims_A
    order = ndims_A - order_inv + 1;
    if order ~= ndims_A
        X{order} = reshape(permute(Gcore{order+1},[1,3,2]),[R(order+1)*size_A(order+1),R(order+2)]) * X{order+1};
        X{order} = reshape(X{order},[R(order+1),prod(size_A(order+1:end))]); % [R(d+1), Id+1 ... I(D)]
    end
end
end