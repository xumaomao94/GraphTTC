function Lambda = update_lambda_GH_graph_ind(A_observed,Gcore,Lambda,Tau,Lap)
    ndims_A = ndims(A_observed);
    size_A = size(A_observed);
    R = zeros(1,ndims_A+1);
    for i = 1:ndims_A+1
        R(i) = length(Lambda.mean{i});
    end

    for i = 2:ndims_A % Lambda{1} and Lambda{ndims_A} are set to 1
        a_alpha = Lambda.a_alpha{i} + Lambda.c/2;
        a_beta = Lambda.a_beta{i} + Lambda.mean{i}./2;
        Lambda.a{i} = a_alpha./a_beta;
        
        Gcoreselfcor_sum_before = zeros(R(i-1),R(i));
        Gcoreselfcor_sum_after = zeros(R(i),R(i+1));
        for p = 1:R(i)
            for q = 1:R(i-1)
                Gcoreselfcor_sum_before(q,p) = trace(Lap{i-1} * (squeeze(Gcore.mean{i-1}(q,p,:))*squeeze(Gcore.mean{i-1}(q,p,:))' + squeeze(Gcore.var{i-1}(q,p,:,:))));
            end
            for q = 1:R(i+1)
                Gcoreselfcor_sum_after(p,q) = trace(Lap{i} * (squeeze(Gcore.mean{i}(p,q,:))*squeeze(Gcore.mean{i}(p,q,:))' + squeeze(Gcore.var{i}(p,q,:,:))));
            end
        end
  
        c = (- 1/2 * size_A(i-1) * R(i-1) - 1/2 * size_A(i) * R(i+1) + Lambda.c)';
        b = (Gcoreselfcor_sum_before' * Lambda.meaninv{i-1}' +  Gcoreselfcor_sum_after * Lambda.meaninv{i+1}' + Lambda.b)';
        
%         Lambda.mean{i} = sqrt(b)./sqrt(Lambda.a{i}) .* besselk(c+1,sqrt(b.*Lambda.a{i}))./  besselk(c,sqrt(b.*Lambda.a{i}));
%         Lambda.meaninv{i} = sqrt(Lambda.a{i})./sqrt(b) .* besselk(c+1,sqrt(b.*Lambda.a{i}))./  besselk(c,sqrt(b.*Lambda.a{i})) -  2.*c./b;
        Lambda.mean{i} = sqrt(b)./sqrt(Lambda.a{i}) .* divbessel(c,sqrt(b.*Lambda.a{i}));
        Lambda.meaninv{i} = sqrt(Lambda.a{i})./sqrt(b) .* divbessel(c,sqrt(b.*Lambda.a{i})) -  2.*c./b;

    end
    Lambda_update = Lambda;

end