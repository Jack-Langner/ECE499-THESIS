
% theta = [omega; alpha; beta; variance45];
%theta = [omega; alpha; beta; var(et)];

%theta = theta(:,1);
function [g,el,s2t,H] = GARCH11_grad_V5(theta,et)
% theta(1) = omega(w)
% theta(2) = alpha(a)
% theta(3) = beta(b)
% theta(4) = sigma^2_1(s21)
% returns the gradient of the log likelihood function wrt to omega, alpha,
% beta and sigma^2_1 evaulated at the point given. and the log likelihood
% of that point. theta is a column vector that contains the parameters

etp = [sqrt(var(et));et].^2;

T = length(et);
s2t = NaN(T+1,1);
s2t(1) = theta(4);
for tt = 2:T+1
    s2t(tt) = theta(1)+theta(2)*etp(tt-1)+theta(3)*s2t(tt-1);
end
e2t = et.^2;
s2t = s2t(2:end);
z2t = e2t./s2t;

Zt = theta(2)*z2t+theta(3);
ft = cumprod(Zt);
Ft = sum(ft.*tril(toeplitz(1./ft)),2);
% s2t = theta(1)*[0;Ft(1:end-1)]+theta(4)*[1;ft(1:end-1)];
el = -0.5*sum(log(s2t)+z2t);

%%
dftda = ft.*cumsum(z2t./Zt);
dftdb = ft.*cumsum(1./Zt);

d2ftda2 = ft.*((cumsum(z2t./Zt)).^2-cumsum((z2t./Zt).^2));
d2ftdb2 = ft.*(cumsum(1./Zt).^2-cumsum((1./Zt).^2));

d2ftdab = ft.*(cumsum(1./Zt).*cumsum(z2t./Zt)-cumsum(z2t./(Zt.^2)));
%
dftda = [0;dftda(1:end-1)];
dftdb = [0;dftdb(1:end-1)];

d2ftda2 = [0;d2ftda2(1:end-1)];
d2ftdb2 = [0;d2ftdb2(1:end-1)];
d2ftdab = [0;d2ftdab(1:end-1)];

tpa = cumsum(z2t./Zt);
mpa = cumsum(tril(toeplitz(z2t./Zt)),2,'reverse');
fp = ft.*tril(toeplitz(1./ft));

dFtda = sum(fp.*(tpa-mpa),2);
dFtda = [0;dFtda(1:end-1)];


tpb = cumsum(1./Zt);
mpb = cumsum(tril(toeplitz(1./Zt)),2,'reverse');
%fp = ft.*tril(toeplitz(1./ft));

tpa22 = cumsum(-(z2t./Zt).^2);
mpa22 = cumsum(tril(toeplitz(-(z2t./Zt).^2)),2,'reverse');
d2Ftda2 = sum(fp.*(((tpa-mpa).^2)+(tpa22-mpa22)),2);
d2Ftda2 = [0;d2Ftda2(1:end-1)];

tpb22 = cumsum(-1./(Zt.^2));
mpb22 = cumsum(tril(toeplitz(-1./(Zt.^2))),2,'reverse');
d2Ftdb2 = sum(fp.*(((tpb-mpb).^2)+(tpb22-mpb22)),2);
d2Ftdb2 = [0;d2Ftdb2(1:end-1)];

tpab2 = cumsum(-z2t./(Zt.^2));
mpab2 = cumsum(tril(toeplitz(-z2t./(Zt.^2))),2,'reverse');
d2Ftdab = sum(fp.*((tril(tpa-mpa).*tril(tpb-mpb))+(tpab2-mpab2)),2);
d2Ftdab = [0;d2Ftdab(1:end-1)];

dFtdb = sum(fp.*(tpb-mpb),2);
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
%end

ds2tda2 = theta(1)*d2Ftda2+theta(4)*d2ftda2;
ds2tdb2 = theta(1)*d2Ftdb2+theta(4)*d2ftdb2;
ds2tdab = theta(1)*d2Ftdab+theta(4)*d2ftdab;

d2eldw2 = sum(-(ds2tdw./s2t).^2+ 2*e2t.*(ds2tdw.^2)./(s2t.^3));
d2eldaw = sum(-(ds2tda.*ds2tdw)./(s2t.^2)+dFtda./s2t-e2t.*((-2*ds2tda.*ds2tdw)./(s2t.^3)+dFtda./(s2t.^2)));
d2eldbw = sum(-(ds2tdb.*ds2tdw)./(s2t.^2)+dFtdb./s2t-e2t.*((-2*ds2tdb.*ds2tdw)./(s2t.^3)+dFtdb./(s2t.^2)));
d2eldsw = sum(-(ds2tds21.*ds2tdw)./(s2t.^2)-e2t.*((-2*ds2tds21.*ds2tdw)./(s2t.^3)));

d2elda2 = sum(-(ds2tda./s2t).^2+ds2tda2./s2t-e2t.*(-2*(ds2tda.^2)./(s2t.^3)+ds2tda2./(s2t.^2)));
d2eldab = sum(-(ds2tdb.*ds2tda)./(s2t.^2)+ds2tdab./s2t-e2t.*((-2*ds2tdb.*ds2tda)./(s2t.^3)+ds2tdab./(s2t.^2)));
d2eldsa = sum(-(ds2tds21.*ds2tda)./(s2t.^2)+dftda./s2t-e2t.*((-2*ds2tds21.*ds2tda)./(s2t.^3)+dftda./(s2t.^2)));

d2eldb2 = sum(-(ds2tdb./s2t).^2+(ds2tdb2./s2t)-e2t.*(-2*(ds2tdb.^2)./(s2t.^3)+ds2tdb2./(s2t.^2)));
d2eldsb = sum(-(ds2tdb.*ds2tds21)./(s2t.^2)+dftdb./s2t-e2t.*(-2*(ds2tdb.*ds2tds21)./(s2t.^3)+dftdb./(s2t.^2)));

d2elds2 = sum(-(ds2tds21./s2t).^2+2*e2t.*(ds2tds21.^2)./(s2t.^3));

H = [d2eldw2 d2eldaw d2eldbw d2eldsw;...
     d2eldaw d2elda2 d2eldab d2eldsa;...
     d2eldbw d2eldab d2eldb2 d2eldsb;...
     d2eldsw d2eldsa d2eldsb d2elds2];
end