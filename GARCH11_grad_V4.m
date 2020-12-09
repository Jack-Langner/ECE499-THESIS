
% theta = [omega; alpha; beta; variance45];
%theta = [omega; alpha; beta; var(et)];


function [g,el,s2t] = GARCH11_grad_V4(theta,et)
% theta(1) = omega(w)
% theta(2) = alpha(a)
% theta(3) = beta(b)
% theta(4) = sigma^2_1(s21)
% returns the gradient of the log likelihood function wrt to omega, alpha,
% beta and sigma^2_1 evaulated at the point given. and the log likelihood
% of that point. theta is a column vector that contains the parameters

%e2t = et.^2;
%z2t = zt.^2;
% Zt = theta(2)*z2t+theta(3);
% ft = cumprod(Zt);
% Ft = sum(ft.*tril(toeplitz(1./ft)),2);


etp = [sqrt(var(et));et].^2;

T = length(et);
s2t = NaN(T+1,1);
s2t(1) = theta(4);
for tt = 2:T+1
    s2t(tt) = theta(1)+theta(2)*etp(tt-1)+theta(3)*s2t(tt-1);
end
e2t = et.^2;
%
s2t = s2t(2:end);
z2t = e2t./s2t;
%z2t = etp(2:end)./s2t(2:end);

Zt = theta(2)*z2t+theta(3);
ft = cumprod(Zt);
Ft = sum(ft.*tril(toeplitz(1./ft)),2);
% s2t = theta(1)*[0;Ft(1:end-1)]+theta(4)*[1;ft(1:end-1)];
el = -0.5*sum(log(s2t)+z2t);
%el = -0.5*sum(log(s2t(2:end))+z2t);
%s2t = s2t(2:end);
% %%
% plot(s2t(:,3))
% hold on
% plot(2:T+1,s2tw(2:end))
% xlim([1 T])
% legend('known','estimated')
% %%
% T=1e3+1;
% plot(1:T-1 , zt.^2)
% hold on
% plot(1:T-1,z2t)
% plot(1:T-1,E2(:,(maxPQ + 1):T)./V(:,(maxPQ + 1):T))
% xlim([1 T-1])
% legend('known','estimated','MW')
%%
dftda = ft.*cumsum(z2t./Zt);
dftdb = ft.*cumsum(1./Zt);

dftda = [0;dftda(1:end-1)];
dftdb = [0;dftdb(1:end-1)];

tp = cumsum(z2t./Zt);
mp = cumsum(tril(toeplitz(z2t./Zt)),2,'reverse');
fp = ft.*tril(toeplitz(1./ft));

dFtda = sum(fp.*(tp-mp),2);
dFtda = [0;dFtda(1:end-1)];

tp = cumsum(1./Zt);
mp = cumsum(tril(toeplitz(1./Zt)),2,'reverse');
fp = ft.*tril(toeplitz(1./ft));

dFtdb = sum(fp.*(tp-mp),2);
dFtdb = [0;dFtdb(1:end-1)];

ds2tdw = [0;Ft(1:end-1)];
deldw = -0.5*sum(ds2tdw./s2t-(e2t.*ds2tdw)./(s2t.^2));

ds2tds21 = [1;ft(1:end-1)];
delds21 = -0.5*sum(ds2tds21./s2t-(e2t.*ds2tds21)./(s2t.^2));

ds2tda = theta(1)*dFtda+theta(4)*dftda;
delda = -0.5*sum(ds2tda./s2t-(e2t.*ds2tda)./(s2t.^2));

ds2tdb = theta(1)*dFtdb+theta(4)*dftdb;
deldb = -0.5*sum(ds2tdb./s2t-(e2t.*ds2tdb)./(s2t.^2));

g = [deldw;delda;deldb;delds21];
end