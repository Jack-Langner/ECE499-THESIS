

function [constant, ar, ma, variance] = initGARCH(et,p,q)

y = (et-mean(et)).^2;
% p = 1;
% q = 1;
n = length(y);
%for p =1:5
m=0:p+q;
rhat = NaN(m(end)+1,1);
for M=0:m(end)
    rhat(M+1)= dot(y(1:(n-M))-mean(y(1:(n-M))),y(M+1:n)-mean(y(M+1:n)))/(n-M); %covariance
end
%ref = [rhat';0:q+p];
ind1 = (q+1:q+p);
ind2 = abs(q:-1:q-p+1)+1;
d2 = rhat(ind1+1);
Rhat = toeplitz(rhat(ind1),rhat(ind2));

ar = (Rhat\d2);
%rah = sort(roots([1 -ar']),'descend');
% [ra rah]
% phihat = [-1; ar];
yp = conv(y,[1 -ar']);
yp(end) = []; %filtered ARMA process so result is MA
%
rphat = NaN(q+1,1);

for jj=0:q
    rphat(jj+1)= dot(yp(1:(n-jj))-mean(yp(1:(n-jj))),yp(jj+1:n)-mean(yp(jj+1:n)))/(n-jj); %covariance
end
%
N = 10; % arbitrary number of convergence cycles
% adapted from Time Series Analysis: Forecasting and Control Third Edition
% by By Box, Jenkins, and Reinsel
tau = NaN(q+1,N+1);
mae = NaN(q+1,N);
tau(:,1) = [sqrt(rphat(1));zeros(q,1)];
f = NaN(q+1,1);
for nn = 2:N+1
for jj = 0:q
    f(jj+1) = tau(1:end-jj,nn-1)'*tau(jj+1:end,nn-1)-rphat(jj+1);
end

T1 = fliplr(triu(toeplitz(flipud(tau(:,nn-1)))));
T2 = triu(toeplitz(tau(:,nn-1)));
T = T1+T2;

tau(:,nn) = tau(:,nn-1) - T\f;
mae(:,nn-1) = [tau(1,nn-1)^2; -tau(2:end,nn-1)/tau(1,nn-1)];
end
% %%
% omega = mean(yp);%mae(1,end)*(1-ar);
% beta = mae(2,end);
% alpha = ar-beta;
% variance45 = var(et);%mae(1,end);

constant = mean(yp);
ma = mae(2:end,end);
variance = mae(1,end);
end