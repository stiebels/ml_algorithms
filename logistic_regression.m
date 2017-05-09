%% Setup

% Generating data
nsamples = 5000;
problem = 'linear';
[feat, label] = construct_data(nsamples,'train',problem); % arbitrary numeric data in format: feat[DxN]; label[1xN]
[feat_test, label_test] = construct_data(nsamples,'test',problem); % arbitrary numeric data in format: feat[DxN]; label[1xN]

% Setting features, labels and initial weights
X = feat';
y = label';
w = rand(size(X,2), 1, 'double');

% Pre-defining some variables
w_prev = rand(size(X,2), 1, 'double');
loss_round = inf;
loss_prev = -inf;

% Set optimizer
opt = 1; % 0 for newton-raphson, 1 for gradient descent

% Set convergence criteria
conv_w = 0.0001;
conv_loss = 0.0001;
alpha = 0.002; % only relevant for gradient descent
lambda = 0.1; % l2 regularization hyperparameter

k = 0;

%% Train

if opt == 0 % newton-raphson
    while (((abs(sum(w/norm(w) - w_prev/norm(w_prev))) > conv_w) | k == 0))
        w_prev = w;

        G = X'*(y-sigmoid(X,w)) - lambda*w;
        R = eye(size(feat,2)).*((sigmoid(X,w) * (1-sigmoid(X,w))'));
        H = X'*R*X + lambda*eye(size(w,1));

        w = w_prev + (H\G);
  
        k = k + 1;
        loss_round(k) = -sum(y.*log(sigmoid(X,w))+(1-y).*log(1-sigmoid(X,w))) + lambda*norm(w)

    end
end

if opt == 1 % gradient descent
    while (((abs(sum(w/norm(w) - w_prev/norm(w_prev))) > conv_w) | k ==0))
        w_prev = w;
        loss_prev = loss_round;
        
        G = X'*(y-sigmoid(X,w)) - lambda*w;
        
        w = w_prev + (alpha*G);
        
        k = k + 1;
        loss_round(k) = -sum(y.*log(sigmoid(X,w))+(1-y).*log(1-sigmoid(X,w))) + lambda*norm(w);
    end
    
end

plot(linspace(0,k, k), loss_round);
title('Training Loss');
xlabel('Updates');
ylabel('Loss');

loss_total = sum(loss_round)

%% Test

X_test = feat_test';
y_test = label_test';
preds = sigmoid(X_test, w);


%% Output

if opt == 0
    disp('Newton-Raphson')
else
    disp('Gradien Descent')
end

disp('Updates until convergence:')
k

disp('METRICS OF THIS RUN:')
loss_total
mea = sum(abs(label_test-preds'))/size(label_test,2) % mean absolute error of probabilities to labels
acc = sum((round(preds) == label_test'))/size(label_test,2) % accuracy
