%% small additive process noise, SV measurement eqn
clear;clc
NN = 100;
phi = 0.8;
muh = 0;
w2h = .0001;
muy = 0;
v = randn(NN,1);

sigma2p = w2h;
%phi = 0.8+sqrt(sigma2p)*randn(NN,1);

h = NaN(NN,1);
h(1) = 1;%muh+sqrt(w2h/(1-phi^2))*randn; 
% can use h(1) for long term volatility targetting, I think
for ii = 2:NN
    h(ii) = muh+phi*(h(ii-1)-muh)+sqrt(w2h)*randn;
end

ymeas = muy+exp(h/2).*v;
%%

%initStateGuess = [0.8; 0.95];
initStateGuess = 0.95;
ukf = unscentedKalmanFilter(@SVukfState,@SVukfMeasNonAdd,initStateGuess,...
       'HasAdditiveMeasurementNoise',false,'HasAdditiveProcessNoise',true);

ukf.ProcessNoise = w2h;%w2h*eye(2);   
%ukf.ProcessNoise = [0 0; 0 w2h];%w2h*eye(2);
ukf.MeasurementNoise = 1;
%ukf.Alpha = 1e0;
%ukf.Kappa = 1e0;


%%
numStates = numel(initStateGuess);
hCorr = NaN(NN,numStates);              %estimate of h[k|k], corrected state estimate
PCorr = NaN(numStates,numStates,NN);    %estimate of P[k|k], corrected state covariance
hPred = NaN(NN,numStates);              %estimate of h[k+1|k], predicted state estimate
PPred = NaN(numStates,numStates,NN);    %estimate of P[k+1|k], predicted state covariance
%innov = NaN(NN,1);                      %keep 1, assuming scalar measurement
%yukf  = NaN(NN,1);
%SVukfMeas(ukf.State)
%%
%m = NaN(NN,1);
%lambda = (0.95).^(NN-1:-1:0);
for k = 1:NN
    %m(k) = (lambda(NN+1-k:NN)*ymeas(1:k))/k; %exponentially weighted average
    %yukf(k) = SVukfMeas(ukf.State(2), m(k)); %UKF guess at y[k]
    %innov(k) = ymeas(k) - yukf(k); %innovation at time k
    [hCorr(k,:), PCorr(:,:,k)] = correct(ukf,ymeas(k));
    [hPred(k,:), PPred(:,:,k)] = predict(ukf,0.8);
end

%%
function x = SVukfState(x,w)
%x(1) = x(1); %phi/AR parameter
x = w*x;%0.12*(1-x(1)) + x(1)*x(2); %0.12 = muh, x(2) is log volatility
end

function y = SVukfMeasNonAdd(x,v)
y = exp(x/2).*v;
end

% function y = SVukfMeas(x,muy)
% y = muy+exp(x/2);
% end