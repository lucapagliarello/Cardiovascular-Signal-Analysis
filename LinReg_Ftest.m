%% Statistics of difference in prediction error variances (conditional entropies) estimated through linear regression
% Uf: residuals of unrestricted (full) regression
% Ur: residuals of restricted regression
% Nf: number of coefficients for full regression (in a VAR model, M*p)
% Nr: number of coefficients for restricted regression (in a VAR model, (M-1)*p)

function [p_value,t_star] = LinReg_Ftest(Uf,Ur,Nf,Nr)

    % F-statistic for significance
    RSSu=sum(Uf.^2);
    RSSr=sum(Ur.^2);
    n1=Nf-Nr; %number of restrictions
    n2=length(Ur)-Nf; % n. of observations - n. tot of coeffs
    t_star=((RSSr-RSSu)/n1)/(RSSu/n2); %test statistics
    p_value=1-cdff_fun(t_star,n1,n2); % function of Seth Toolbox

end


%% function of Seth Toolbox: CDF computes the cumulative 'F' distribution
function p = cdff_fun(x,v1,v2)
    p = 0;
    t = (v1 <= 0 | v2 <= 0 | isnan(x) | isnan(v1) | isnan(v2));
    p(t) = NaN;
    s = (x==Inf) & ~t;
    if any(s)
       p(s) = 1;
       t = t | s;
    end

    % Compute P when X > 0.
    k = find(x > 0 & ~t & isfinite(v1) & isfinite(v2));
    if any(k)
        xx = x(k)./(x(k) + v2(k)./v1(k));
        p(k) = betainc(xx, v1(k)/2, v2(k)/2);
    end

    if any(~isfinite(v1(:)) | ~isfinite(v2(:)))
       k = find(x > 0 & ~t & isfinite(v1) & ~isfinite(v2) & v2>0);
       if any(k)
          p(k) = chi2cdf(v1(k).*x(k),v1(k));
       end
       k = find(x > 0 & ~t & ~isfinite(v1) & v1>0 & isfinite(v2));
       if any(k)
          p(k) = 1 - chi2cdf(v2(k)./x(k),v2(k));
       end
       k = find(x > 0 & ~t & ~isfinite(v1) & v1>0 & ~isfinite(v2) & v2>0);
       if any(k)
          p(k) = (x(k)>=1);
       end
    end
end