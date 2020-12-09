% Redoing GARCH(1,1) in light of loglikely.pdf from Fred, didnt calculate
% the derivatives for gradient ascent correctly.

%% s^2_t = w+ae^2_{t-1}+b*s^2_{t-1} aka GARCH(1,1)
%T = 20;
T = 1e3;
t = (1:T).';
w = 0.15;
a = 0.1;
b = 0.7;
s21 = 0.7;
mu = 3;

THETA = [w,a,b,s21];
zt = randn(T,1);
et = NaN(T,1);
rt = NaN(T,1);
et(1) = sqrt(s21)*zt(1);
%rt(1) = mu+et(1);
s2t = NaN(T,3);
s2t(1,1) = s21;
for tt = 2:T
    s2t(tt,1) = w+a*et(tt-1).^2+b*s2t(tt-1,1);
    et(tt) = sqrt(s2t(tt,1))*zt(tt);
    %rt(tt) = mu+et(tt);
end
z2t = zt.^2;
e2t = et.^2;
% rt = mu+et;

bpt = b.^(t-1);
B = [0;cumsum(b.^((0:T-2).'))];
P = tril(toeplitz([0;b.^((0:T-2).')]))*e2t;
s2t(:,2) = w*B+a*P+bpt*s21;
clear bpt B P tt mu

Zt = a*z2t+b;
ft = cumprod(Zt);
Ft = sum(ft.*tril(toeplitz(1./ft)),2);
s2t(:,3) = w*[0;Ft(1:end-1)]+s21*[1;ft(1:end-1)];

% ft = [1;cumprod(Zt(1:end-1))];
% Ft = [0;sum(ft(2:end).*tril(toeplitz(1./ft(2:end))),2)];
% s2t(:,3) = w*Ft+s21*ft;
el_known = -0.5*sum(log(s2t)+(e2t./s2t));
clear z2t e2t Zt a b s21 w
%mean(s2t(:,3))
%% check MATLAB garch estimate
%Mdl = garch(1,1);
T = 1e3;
% Mdl = garch('Constant',0.0001,'GARCH',0.5,'ARCH',0.2);
% [v,y] = simulate(Mdl,T);
%The output v contains simulated conditional variances. y is a column vector of simulated responses (innovations).

%Specify a GARCH(1,1) model with unknown coefficients, and fit it to the series y.

ToEstMdl = garch(1,1);

%[EstMdl, EstParamCov,logL,info] = estimate(ToEstMdl,et,'Display','iter');
[EstMdl, EstParamCov,logL,info] = estimate(ToEstMdl,et);
el_MW = logL+T*log(2*pi)/2;

[g,el,~] = GARCH11_grad_V4([EstMdl.Constant;cell2mat(EstMdl.ARCH);cell2mat(EstMdl.GARCH);var(et)],et)
%%
[omega, ar, ma] = initGARCH(et,1,1);
beta = ma;
alpha = ar-ma;
%% Using V4
N = 5000;%1250*8;%8750
stepSize = 5e-5;
% e2t = et.^2;
% N = 10;
% stepSize = 1e-3;
theta = NaN(4,N+1);
theta(:,1) = [omega; alpha; beta; var(et)];
%theta(:,1) = theta(:,end);
Grad = NaN(4,N+1);
magG = NaN(N+1,1);
EL = NaN(N+1,1);

tic
for ii = 1:N
    [g,el,~] = GARCH11_grad_V4(theta(:,ii),et);
    theta(:,ii+1) = theta(:,ii)+stepSize*g/sqrt(g'*g);
    Grad(:,ii) = g;
    magG(ii) = sqrt(g'*g);
    EL(ii) = el;
end

[g,el,~] = GARCH11_grad_V4(theta(:,end),et);
Grad(:,end) = g;
magG(end) = sqrt(g'*g);
EL(end) = el;
toc

%%
%theta = theta2;
for plm = 1
figure
subplot(2,3,1)
plot(theta(1,:))
hold on
plot([0 N],[THETA(1) THETA(1)])
xlabel('\omega')

subplot(2,3,2)
plot(theta(2,:))
hold on
plot([0 N],[THETA(2) THETA(2)])
xlabel('\alpha')

subplot(2,3,4)
plot(theta(3,:))
hold on
plot([0 N],[THETA(3) THETA(3)])
xlabel('\beta')

subplot(2,3,5)
plot(theta(4,:))
hold on
plot([0 N],[THETA(4) THETA(4)])
xlabel('\sigma^2_1')

subplot(2,3,[3 6])
plot((0:N).',EL)
hold on
plot([0 N],[el_known(1) el_known(1)])
plot([0 N],[el_MW el_MW])
title('ML')

figure
plot(0:N, log(magG))
title('log(|g|)')
end
