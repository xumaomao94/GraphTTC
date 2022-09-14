function y = divbessel(v,x)
% calculate besselk(v+1,x) / besselk(v,x)

    if abs(v)<=2
        zero_bessel = ismember(besselk(v,x),0);
        index_zero_bessel = find(zero_bessel);
        
        if isempty(index_zero_bessel) % no zero value in the denominator
            y = besselk(v+1,x)./besselk(v,x);
        elseif sum(x(index_zero_bessel) >= 600) == length(index_zero_bessel) % which means that all 0 are caused by large x
            y = besselk(v+1,x)./besselk(v,x);
            y(index_zero_bessel) = exp(-1);
        else
            error('unknown case for calculating the bessel function')
            
        end
    else
        if v >= 0
            y = 2.*v./x + 1./divbessel(v-1,x);
        else
            y = 1./( divbessel(v+1,x) - 2.*(v+1)./x);
        end
    end
end