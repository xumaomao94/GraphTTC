function [E,T] = update_ET_ind(A_observed,Mask,Tau,W_final,E,T)

    W_final = reshape(W_final,size(A_observed));
    E.var = Tau.mean * Mask + T.mean;
    E.var = 1./E.var;    
    E.mean = Tau.mean*Mask.*E.var.*( A_observed - W_final );

    alpha = T.a + 1/2;
    beta = T.b + (E.mean.^2 + E.var)./2;
    T.mean = alpha./beta;
end