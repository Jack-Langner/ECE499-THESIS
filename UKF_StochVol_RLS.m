%% small additive process noise, SV measurement eqn
clear;clc
N = 1e3;
phi = 0.8;
muh = 0.12;
w2h = .0001;
muy = 0;
v = randn(N,1);

sigma2p = w2h;
%phi = 0.8+sqrt(sigma2p)*randn(NN,1);

initVar = 0.5;
h = NaN(N,1);
h(1) = initVar;%muh+sqrt(w2h/(1-phi^2))*randn;
hn = NaN(N,1);
hn(1) = initVar;
% can use h(1) for long term volatility targetting, I think
for k = 2:N
    h(k) = muh+phi*(h(k-1)-muh)+sqrt(w2h)*randn;
    hn(k) = muh+phi*(hn(k-1)-muh);
end


ymeas = muy+exp(h/2).*v;
%
%initStateGuess = [0.8; 0.95];
initStateGuess = 0.5;
ukf = unscentedKalmanFilter(@SVukfState,@SVukfMeasNonAdd,initStateGuess,...
       'HasAdditiveMeasurementNoise',false,'HasAdditiveProcessNoise',true);

ukf.ProcessNoise = w2h;%w2h*eye(2);   
%ukf.ProcessNoise = [0 0; 0 w2h];%w2h*eye(2);
ukf.MeasurementNoise = 1;

ukf2 = clone(ukf);
%ukf.Alpha = 1e0;
%ukf.Kappa = 1e0;
%
numStates = numel(initStateGuess);
hCorr = NaN(N,numStates);              %estimate of h[k|k], corrected state estimate
PCorr = NaN(numStates,numStates,N);    %estimate of P[k|k], corrected state covariance
hPred = NaN(N,numStates);              %estimate of h[k+1|k], predicted state estimate
PPred = NaN(numStates,numStates,N);    %estimate of P[k+1|k], predicted state covariance
%innov = NaN(NN,1);                      %keep 1, assuming scalar measurement
%yukf  = NaN(NN,1);
%SVukfMeas(ukf.State)

RLSstruct.p = 1;
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

prar = 1;
rar = recursiveAR(prar,[1 0],'ForgettingFactor',1);

Arar = NaN(N,prar+1);
EstOut = NaN(N,1);

% for ii = 1:N
%     [Arar(ii,:), EstOut(ii)] = rar(y(ii+prar));
% end
%

for k = 1:N
    u = ukf.State;
    %m(k) = (lambda(NN+1-k:NN)*ymeas(1:k))/k; %exponentially weighted average
    %yukf(k) = SVukfMeas(ukf.State(2), m(k)); %UKF guess at y[k]
    %innov(k) = ymeas(k) - yukf(k); %innovation at time k
    [hCorr(k,:), PCorr(:,:,k)] = correct(ukf,ymeas(k));
    [Arar(k,:),~] = rar(hCorr(k,:));
    
    
    %u = ukf.State;
    RLSstruct.Pi(:,k) = RLSstruct.Li*RLSstruct.P(:,:,k)*u;
    RLSstruct.k(:,k) = RLSstruct.Pi(:,k)/(1+u'*RLSstruct.Pi(:,k));
    RLSstruct.xi(k) = hCorr(k,:) - RLSstruct.w(:,k)'*u;
    RLSstruct.w(:,k+1) = RLSstruct.w(:,k) + RLSstruct.k(:,k)*RLSstruct.xi(k);
    RLSstruct.P(:,:,k+1) = RLSstruct.Li*RLSstruct.P(:,:,k) - ...
        RLSstruct.Li*RLSstruct.k(:,k)*u'*RLSstruct.P(:,:,k);
    [hPred(k,:), PPred(:,:,k)] = predict(ukf,RLSstruct.w(:,k+1));
end

%%
figure
subplot(1,2,1)
plot([hCorr hPred h hn]);
legend('Corr','Pred','true','true, no noise')
subplot(1,2,2)
plot(0:N,RLSstruct.w')
hold on
plot([0 N],[phi phi])
legend('RLS','true','location','best')
%%
function x = SVukfState(x,w)
%x(1) = x(1); %phi/AR parameter
x = 0.15*(1-w) + w*x; %0.12 = muh, x(2) is log volatility
end

function y = SVukfMeasNonAdd(x,v)
y = exp(x/2).*v;
end
