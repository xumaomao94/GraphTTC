function [Gcore,X] = als_core(A_observed,S,Gcore,Lap,beta,X)
    
size_A = size(A_observed);
ndims_A = ndims(A_observed);
R = zeros(1,ndims_A+1); R(1) = 1; R(end) = 1;
for i = 2:ndims_A
    R(i) = size(Gcore{i},1);
end

% the update for the order-th TTcore
W = cell(1,ndims_A); W{1} = 1; % to store the mean's product before the current core, actually only on W is enough, no need to store all W{d}

for order = 1:ndims_A
    if order ~= 1
        W{order} = W{order-1} * reshape(Gcore{order-1},R(order-1),R(order)*size_A(order-1));
        W{order} = reshape(permute(reshape(W{order},[prod(size_A(1:order-2)),R(order),size_A(order-1)]),[1,3,2]),[prod(size_A(1:order-1)),R(order)]); % J1,J2,...,Jd-1 * Rd
    end
    
    A_order = reshape(permute(reshape(A_observed,[prod(size_A(1:order-1)),size_A(order),prod(size_A(order+1:end))]), [1,3,2]),[prod(size_A(1:order-1))*prod(size_A(order+1:end)),size_A(order)]);  % [I1...I(d-1) I(d+1)...I(D) , I(d)]
    O_order = reshape(permute(reshape(S,[prod(size_A(1:order-1)),size_A(order),prod(size_A(order+1:end))]), [1,3,2]),[prod(size_A(1:order-1))*prod(size_A(order+1:end)),size_A(order)]);  % [I1...I(d-1) I(d+1)...I(D) , I(d)]
    
    Upsilon = zeros(size_A(order)*R(order)*R(order+1));
    mu = zeros([size_A(order)*R(order)*R(order+1),1]);
    for d = 1:size_A(order)
        X_kron_Wt = kron(X{order},W{order}'); % [R(d) R(d+1),I1...I(d-1) I(d+1)...I(D)]
        O_khatri_XWt = khatrirao( O_order(:,d)', X_kron_Wt ); % [R(d) R(d+1),I1...I(d-1) I(d+1)...I(D)]
        Upsilon((d-1)*R(order)*R(order+1)+1:d*R(order)*R(order+1),(d-1)*R(order)*R(order+1)+1:d*R(order)*R(order+1)) = O_khatri_XWt * O_khatri_XWt';% [R(d) R(d+1),R(d) R(d+1),]
        
        mu((d-1)*R(order)*R(order+1)+1:d*R(order)*R(order+1)) = O_khatri_XWt * (O_order(:,d).*A_order(:,d));
        
    end
    Upsilon = Upsilon + kron(beta(order)*Lap{order},eye(R(order)*R(order+1)));
    
    sol = Upsilon\mu; % [R(d)*R(d+1)*J(d),1]
    Gcore{order} = reshape(sol,[R(order),R(order+1),size_A(order)]);
end


end