%%
% Compare performance of UKF, LMS, RLS with an AR(p) model.

%%
clear;clc
N = 1e3;
s2 = .01;
v = sqrt(s2)*randn(N,1);
%v = (1:N)';

p = 2;
%w = rand(p,1);
r = [0.8; -0.354];
arp = poly(r);
w = (-arp(2:end)).';
%w = [0.8];
%w = ones(p,1);
c = 0;
%
y = NaN(N+p,1);
y(1:p) = zeros(p,1);
%t = (1-p:N)';

for ii = 1:N
    y(ii+p) = c+(w')*y(ii+p-1:-1:ii)+v(ii);
end
%%
y1 = filter(1,arp,v);
%% LMS
LMSstruct.p = 2;
LMSstruct.mu = 100e-3;
LMSstruct.w = NaN(LMSstruct.p,N);
LMSstruct.w(:,1) = zeros(LMSstruct.p,1);
LMSstruct.err = NaN(N,1);
LMSstruct.y = NaN(N,1);

% plms = 2;
% mu = 15e-3;
% wlms = NaN(plms,N+1);
% wlms(:,1) = zeros(plms,1);
% elms = NaN(N,1);

for ii = 1:N
    LMSstruct.y(ii) = (LMSstruct.w(:,ii))'*y(ii+LMSstruct.p-1:-1:ii);
    LMSstruct.err(ii) = y(ii+LMSstruct.p)-LMSstruct.y(ii);
    LMSstruct.w(:,ii+1) = LMSstruct.w(:,ii)+...
        LMSstruct.mu*LMSstruct.err(ii)*y(ii+LMSstruct.p-1:-1:ii);
    
%     ylms = (wlms(:,ii)')*y(ii+plms-1:-1:ii);
%     elms(ii) = y(ii+plms)-ylms;
%     wlms(:,ii+1) = wlms(:,ii)+mu*elms(ii)*y(ii+plms-1:-1:ii);
end

%% NLMS (normalized LMS)

%% RLS
RLSstruct.p = 2;
RLSstruct.delta = 1;
RLSstruct.L = 1; %lambda
RLSstruct.Li = 1/RLSstruct.L;
RLSstruct.w = NaN(RLSstruct.p,N+1);
RLSstruct.w(:,1) = zeros(RLSstruct.p,1);
RLSstruct.P = NaN(RLSstruct.p,RLSstruct.p,N+1);
RLSstruct.P(:,:,1) = RLSstruct.delta*eye(RLSstruct.p);
RLSstruct.Pi = NaN(RLSstruct.p,N);
RLSstruct.k = NaN(RLSstruct.p,N);
RLSstruct.xi = NaN(N,1);
RLSstruct.err = NaN(N,1);

% prls = 2;
% delta = 1;
% L = 1;
% Li = 1/L;
% 
% wrls = NaN(prls,N+1);
% wrls(:,1) = zeros(prls,1);
% 
% Prls = NaN(prls,prls,N+1);
% Prls(:,:,1) = delta*eye(2);
% Pirls = NaN(prls,N);
% krls = NaN(prls,N);
% xi_rls = NaN(N,1);
% erls = NaN(N,1);

for ii = 1:N
    u = y(ii+RLSstruct.p-1:-1:ii);
    
    RLSstruct.Pi(:,ii) = RLSstruct.Li*RLSstruct.P(:,:,ii)*u;
    RLSstruct.k(:,ii) = RLSstruct.Pi(:,ii)/(1+u'*RLSstruct.Pi(:,ii));
    RLSstruct.xi(ii) = y(ii+RLSstruct.p) - RLSstruct.w(:,ii)'*u;
    RLSstruct.w(:,ii+1) = RLSstruct.w(:,ii) + RLSstruct.k(:,ii)*RLSstruct.xi(ii);
    RLSstruct.P(:,:,ii+1) = RLSstruct.Li*RLSstruct.P(:,:,ii) - ...
        RLSstruct.Li*RLSstruct.k(:,ii)*u'*RLSstruct.P(:,:,ii);
    
%     Pirls(:,ii) = Li*Prls(:,:,ii)*u;
%     krls(:,ii) = Pirls(:,ii)/(1+u'*Pirls(:,ii));
%     xi_rls(ii) = y(ii+prls)-wrls(:,ii)'*u;
%     wrls(:,ii+1) = wrls(:,ii)+krls(:,ii)*xi_rls(ii);
%     Prls(:,:,ii+1) = Li*Prls(:,:,ii)-Li*krls(:,ii)*u'*Prls(:,:,ii);
end
%% 
prar = 2;
rar = recursiveAR(prar,[1 0 0],'ForgettingFactor',1);

Arar = NaN(N+1,prar+1);
EstOut = NaN(N+1,1);

for ii = 1:N
    [Arar(ii,:), EstOut(ii)] = rar(y(ii+prar));
end

%%
subplot(1,2,1)
plot(-Arar(:,2))
hold on
plot([1 N],[w(1) w(1)]);
subplot(1,2,2)
plot(-Arar(:,3))
hold on
plot([1 N],[w(2) w(2)]);
%%
prls = 2;
rlsObj = recursiveLS(prls,[0 0]);

wrls = NaN(N,prls);

for ii = 1:N
    u = y(ii+prls-1:-1:ii);
    wrls(ii,:) = rlsObj(y(ii+prls),u);
end
subplot(1,2,1)
plot(wrls(:,1))
hold on
plot([1 N],[w(1) w(1)]);
subplot(1,2,2)
plot(wrls(:,2))
hold on
plot([1 N],[w(2) w(2)]);
%% UKF
pukf = 2;
wukf = NaN(pukf,N+1);
wukf(:,1) = zeros(pukf,1); % initial state guess
ukf = unscentedKalmanFilter(@ARState,@ARMeas,wukf(:,1),...
       'HasAdditiveMeasurementNoise',true,'HasAdditiveProcessNoise',true);
   
ukf.ProcessNoise = 0;%w2h*eye(2);
ukf.MeasurementNoise = s2;

wCorr = NaN(pukf,N);         %estimate of h[k|k], corrected state estimate
PCorr = NaN(pukf,pukf,N);    %estimate of P[k|k], corrected state covariance
wPred = NaN(pukf,N);         %estimate of h[k+1|k], predicted state estimate
PPred = NaN(pukf,pukf,N);    %estimate of P[k+1|k], predicted state covariance
innov = NaN(N,1);            %keep 1, assuming scalar measurement
yukf  = NaN(N,1);

for ii = 1:N
    u = y(ii+pukf-1:-1:ii);
    yukf(ii) = ARMeas(ukf.State,u); %UKF guess at y[k]
    innov(ii) = y(ii+pukf) - yukf(ii); %innovation at time k
    [wCorr(:,ii), PCorr(:,:,ii)] = correct(ukf,y(ii+pukf),u);
    [wPred(:,ii), PPred(:,:,ii)] = predict(ukf);
end
%% Plotting
subplot(1,2,1)
plot([1 N],[w(1) w(1)]);
hold on
plot(LMSstruct.w(1,:)')
plot(RLSstruct.w(1,:)')
plot(wCorr(1,:)')
legend('true','LMS','RLS','UKF','location','best')

subplot(1,2,2)
plot([1 N],[w(2) w(2)]);
hold on
plot(LMSstruct.w(2,:)')
plot(RLSstruct.w(2,:)')
plot(wCorr(2,:)')
legend('true','LMS','RLS','UKF','location','best')
%% Plotting with Matlab RLS
subplot(1,2,1)
plot([1 N],[w(1) w(1)]);
hold on
plot(LMSstruct.w(1,:)')
plot(RLSstruct.w(1,:)')
plot(wCorr(1,:)')
plot(-Arar(:,2))
plot(wrls(:,1))
legend('true','LMS','RLS','UKF','RecursiveAR','RecursiveLS','location','best')

subplot(1,2,2)
plot([1 N],[w(2) w(2)]);
hold on
plot(LMSstruct.w(2,:)')
plot(RLSstruct.w(2,:)')
plot(wCorr(2,:)')
plot(-Arar(:,3))
plot(wrls(:,2))
legend('true','LMS','RLS','UKF','RecursiveAR','RecursiveLS','location','best')
%% UKF function defs
function y = ARMeas(w,u)
y = w'*u;
end

function w = ARState(w)
w = w;
end
