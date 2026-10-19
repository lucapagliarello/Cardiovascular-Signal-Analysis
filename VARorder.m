%% Finds optimal VAR model order (Akaike Information Criterion and Bayesian Information Criterion)

%%% INPUT
% data: original time series (each time series in a column)
% jv: indexes of predicted series
% iv: indexes of predictors
% iv_lags: lagsof predictors, in a row vector (they are all the same for the various predictors)

function out=VARorder(data,pmax)

% pmax=10;
[N,M]=size(data);

% figures of merit
aic=NaN*ones(pmax,1); bic=aic; detSu=aic;
for p=1:pmax   
    ret=LinReg(data,(1:M),(1:M),(1:p)); 
    detSu(p)=det(ret.es2u);
    %formula multivariate AIC 
    aic(p)=log(detSu(p))+2*M*M*p/N; % S matrice di covarianza
    %formula multivariate BIC
    bic(p)=log(detSu(p))+log(N)*M*M*p/N; % S matrice di covarianza
end
% plot(aic);hold on; plot(bic)
pottaic=find(aic == min(aic));
pottbic=find(bic == min(bic));

out.pottaic=pottaic;
out.pottbic=pottbic;
out.aic=aic;
out.bic=bic;
out.detSu=detSu;

end
