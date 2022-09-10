function [Gcore,Lambda] = rank_reduce_relative_GH_graph_ind_Power(ndims_A,Gcore,Lambda,threshold)

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
    Power = cell(1,ndims_A+1);
    Power{1} = Power_L{1};
    Power{ndims_A+1} = Power_H{ndims_A};
    for order = 2:ndims_A
       Power{order} = Power_L{order} + Power_H{order-1};
    end

    ndims_A = length(Gcore.mean);
    for i = 1:ndims_A
        Gcore_mean_temp = Gcore.mean{i};
        Gcore_var_temp = Gcore.var{i};

%         Gcore_cor_temp = Gcore.cor{i};
%         Gcore_kron_temp = Gcore.kron{i};
        lambdainv_row_temp = Lambda.meaninv{i};
        lambdainv_column_temp = Lambda.meaninv{i+1};

        lambda_row_temp = Lambda.mean{i};
        a_temp = Lambda.a{i};
        aalpha_temp = Lambda.a_alpha{i};
        abeta_temp = Lambda.a_beta{i};
        
        row_discarded = Power{i} < threshold;
        ind_row_discarded = find(row_discarded);
        column_discarded = Power{i+1} < threshold;
        ind_column_discarded = find(column_discarded);
        
        
        % eliminate rows and columns from the Gcore
        % -- mean --
        Gcore_mean_temp(ind_row_discarded,:,:) = [];
        Gcore_mean_temp(:,ind_column_discarded,:) = [];
        Gcore_var_temp(ind_row_discarded,:,:,:) = [];
        Gcore_var_temp(:,ind_column_discarded,:,:) = [];

       
        % eliminate elements in Lambda, only operate on the lambda_row, since the lambda_column are needed in the next rank reduction
        % -- mean --
        lambdainv_row_temp(ind_row_discarded) = [];
        lambda_row_temp(ind_row_discarded) = [];
        a_temp(ind_row_discarded) = [];
        aalpha_temp(ind_row_discarded) = [];
        abeta_temp(ind_row_discarded) = [];
        
        % substitute the structure with temp
        Gcore.mean{i} = Gcore_mean_temp;
        Gcore.var{i} = Gcore_var_temp;

%         Gcore.cor{i} = Gcore_cor_temp;
%         Gcore.kron{i} = Gcore_kron_temp;
        Lambda.mean{i} = lambda_row_temp;
        Lambda.meaninv{i} = lambdainv_row_temp;
        Lambda.a{i} = a_temp;
        Lambda.a_alpha{i} = aalpha_temp;
        Lambda.a_beta{i} = abeta_temp;
    end



end