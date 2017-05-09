% Simple Neural Network with 2 input nodes, 1 hidden layer with 2 hidden
% units, 1 output node. -- UNDER DEVELOPMENT --

clear all;

hidden = 2; % not flexible yet
output = 1; % not flexible yet

feat = [0,0;-1,-1;0,1;-1,0]';
V = rand(output, hidden, 'double')';
W = rand(size(feat,1), hidden, 'double');
label = [0;0;1;1];

alpha = 0.1; % learning rate
epochs = 1000;

for epoch = 1:epochs
    
    for i = 1:size(feat,2) 
        
        % For now only batch_size = 1
        x = feat(:,i);
        t = label(i);
        
        % Forward propagation
        a = W*x;
        b = 1./(1+exp(-a));
        z = V'*b;
        y = 1./(1+exp(-z));


        % Backpropagation

        % Output Layer
        dldy = (t-y);
        dydz = (1-y)'*y;
        net_out = dldy*dydz;
        dzdv = b;
        V = V + alpha*(net_out*dzdv);

        % Hidden Layer
        dzdb = V;
        dbda = ((1-b).*b)';
        dadw = x;
        W = W + alpha*(net_out*dzdb.*dbda'*dadw');

        loss_sample(i) = (1/2)*(t-y)^2;
        
    end
    loss_epoch(epoch) = sum(loss_sample);
    
end

plot(linspace(0,epochs, epochs), loss_epoch);
title('Training Loss');
xlabel('Epoch');
ylabel('Loss');

test_feat = [0,0;-1,-1;0,1;-1,0]';
test_label = [0;0;1;1];

for j = 1:size(test_feat,2)
    
    x = test_feat(:,j);
    
    a = W*x;
    b = 1./(1+exp(-a));
    z = V'*b;
    preds(j) = 1./(1+exp(-z));
    
end

disp('METRICS OF THIS RUN:')
loss_total = sum(loss_epoch)
mea = sum(abs(test_label-preds'))/size(test_label,1) % mean absolute error of probabilities to labels
acc = sum((round(preds) == test_label'))/size(test_label,1) % accuracy
