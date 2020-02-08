function [yn] = neterfit(inputs, targets, HidenandOutputLayersizeC, ActivationFuncs, Xn)
%neterfit fiting ANN and eq function  
%   using forward newton vanilla neural network

% Create a Fitting Network

hiddenLayerSize = HidenandOutputLayersizeC ;%[2, 2];

TF = ActivationFuncs ; %{'tansig', 'purelin', 'purelin'};
net = newff(inputs,targets,hiddenLayerSize,TF);

% Input and Output Pre/Post-Processing Functions

net.inputs{1:end}.processFcns = {'removeconstantrows'}; % ,'mapminmax'};
net.outputs{1:end}.processFcns = {'removeconstantrows'}; % ,'mapminmax'};


% Division of Data for Training, Validation, Testing

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 10/100;

% training function 

net.trainFcn = 'trainlm';  % Levenberg-Marquardt

% Error Func

net.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions

net.plotFcns = {'plotperform','ploterrhist','plotregression','plotfit'};


net.trainParam.showWindow=false;
net.trainParam.showCommandLine=false;
net.trainParam.show=100;
net.trainParam.epochs=1000;
net.trainParam.goal=1e-9;
net.trainParam.max_fail=200;



% Train the Network

[net,tr] = train(net,inputs,targets);

% Evaluate the Network

outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

% Recalculate Training, Validation and Test Performance

trainInd=tr.trainInd; % train data index
trainInputs = inputs(:,trainInd);
trainTargets = targets(:,trainInd);
trainOutputs = outputs(:,trainInd);
trainErrors = trainTargets-trainOutputs;
trainPerformance = perform(net,trainTargets,trainOutputs);

valInd=tr.valInd;
valInputs = inputs(:,valInd);
valTargets = targets(:,valInd);
valOutputs = outputs(:,valInd);
valErrors = valTargets-valOutputs;
valPerformance = perform(net,valTargets,valOutputs);

testInd=tr.testInd;
testInputs = inputs(:,testInd);
testTargets = targets(:,testInd);
testOutputs = outputs(:,testInd);
testError = testTargets-testOutputs;
testPerformance = perform(net,testTargets,testOutputs);

% Results for Target #1
% PlotResults(targets(1,:),outputs(1,:),'All Data (1)');
% PlotResults(trainTargets(1,:),trainOutputs(1,:),'Train Data (1)');
% PlotResults(valTargets(1,:),valOutputs(1,:),'Validation Data (1)');
% PlotResults(testTargets(1,:),testOutputs(1,:),'Test Data (1)');


% View the Network
% view(net);

% Plots
% Uncomment these lines to enable various plots.

figure;
plotperform(tr);

% figure;
% plottrainstate(tr);

% figure;
% plotfit(net,inputs,targets);

figure;
plotregression(trainTargets,trainOutputs,'Train Data',...
    valTargets,valOutputs,'Validation Data',...
    testTargets,testOutputs,'Test Data',...
    targets,outputs,'All Data')


% figure;
% ploterrhist(errors);


BI = net.B;% bias
InputW = cell2mat(net.IW);% input weights
LayersW = {};% other weights
LsW = net.LW;

 for i = 1:(length(LsW))^2

      if ~isempty(cell2mat(LsW(i)))

         LayersW(end+1) = LsW(i);
      end
 end
 
 FirstFunc = str2func(TF{1});
 y0 = FirstFunc(BI{1} + InputW * Xn');
 yn = y0;
 for i =1:length(hiddenLayerSize)
    func_n = str2func(TF{i+1});
    
    yn =func_n( BI{i+1} + cell2mat(LayersW(i)) *yn );
 end


end

