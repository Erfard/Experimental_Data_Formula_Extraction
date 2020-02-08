%For Testing perpose and User Guide of Neterfit Fucntion
%  this will generate some test data and get tuned function

% Input Variables Name
syms phi distvar distmean
Xn = [phi, distvar, distmean]; % exampls 

% Sample data for network can be distribution Parameters
x1 = (randi([-1000,1000],100,1));
x2 = (x1.^2+abs(x1))/1e6;
x3 = x2.^0.5-abs(x1).^0.5;
y = (cos(2^0.5*x1-x2)./(sin(x3*5^0.5)+(randn(100,1)-0.5)/20));
y = y.*min(abs(y));
y = y./max(y)+(x1-x2+x3)/1000;
x1 = x1/1000;
x3 = x3/31;

%data complexity and relation plots 
plot(x1, y)
hold on
plot(x2, y)
hold on
plot(x3, y)
legend(string(sym2cell(Xn)))


%Net Training and Experimental Formula Extraction
yn = neterfit([x1'; x2'; x3'], y', [2, 2], {'tansig', 'logsig', 'purelin'}, Xn);
pretty(yn)

