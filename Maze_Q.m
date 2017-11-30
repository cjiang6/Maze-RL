%%%%%%%%%%%%%%%%%%%%% Q-learning for maze navigation %%%%%%%%%%%%%%%%%%%%


clear all
close all
clc

%% Simulation setup
% Environment and Robot
numCols = 20;   % cloumn number of the maze
numRows = 20;   % row number of the maze
numSt = numCols * numRows; % number of states
goal = [numCols-1,numRows-1];   % define goal position (index)

numAct = 4;     % number of actions
initSt = [0,0]; % initial robot position
UP = 1;     % action    
DOWN = 2;
LEFT = 3;
RIGHT = 4;

% Parameters for the algorithm
gamma = 0.99;         % discount factor
alpha = 0.01;		  % initial learning rate
epsi = 0.01;          % inital probability of random action (epsilon) 

numEpi = 10000;        % number of episodes
numStep = 1000;       % number of steps in each episode

% Initialize Q-Table
%Qtable = 10*rand(numCols,numRows,numAct)-5;
Qtable = zeros(numCols,numRows,numAct);
QtableOpt = zeros(numCols,numRows);

reward = [100,-10,-1];    % define reward function ([goal,out-of-bound,otherwise])

% Lists of some variables
rList = [];    % total reward over episodes
dList = [];    % indicator of reaching goal over episodes 
jList = [];    % total steps over episodes
outList = [];  % indicator of out-of-bound over episodes

%% Learning process
for i = 1:numEpi
    sNew = initSt;    
    rAll = 0;
    j = 0;
    flagOut = 0;
    flagDest = 0;
    while j < numStep
        j = j+1; 
        s = sNew;

        if rand < epsi
            a = randi(4);            
        else
            [val,indx] = max(Qtable(s(1)+1,s(2)+1,:));
            a = indx;
        end       
              
        % update robot state by taking action a
        switch a
            case UP
                sNew = s + [0,1];
            case DOWN
                sNew = s + [0,-1];
            case LEFT
                sNew = s + [-1,0];
            case RIGHT
                sNew = s + [1,0];
        end
        
        % calculate TD target
        if sNew == goal
            r = reward(1);
            targTD = r + gamma*max(Qtable(sNew(1)+1,sNew(2)+1,:)); % TD target
        elseif sNew(1) > numCols-1 || sNew(1) < 0 || sNew(2) > numRows-1 || sNew(2) < 0
            r = reward(2);
            targTD = nan;
            flagOut = 1;
        else
            r = reward(3);
            targTD = r + gamma*max(Qtable(sNew(1)+1,sNew(2)+1,:)); % TD target
        end
        
        % update Q-Table
        Qtable(s(1)+1,s(2)+1,a) = Qtable(s(1)+1,s(2)+1,a) + alpha*(targTD - Qtable(s(1)+1,s(2)+1,a));
        rAll = rAll + r;
      
        if sNew == goal
            flagDest = 1;
        end        
              
        if flagOut == 1 || flagDest == 1
            break;
        end
        
    end
    
    rList = [rList, rAll];
    dList = [dList, flagDest];
    jList = [jList, j];
    outList = [outList, flagOut];
    
end

for ii = 1:numCols
    for jj = 1:numRows
        QtableOpt(ii,jj) = max(Qtable(ii,jj,:));
    end
end

%% Plot figures
% Plot total reward of each episode
figure(1)
plot(rList)
xlabel('Episodes')
ylabel('Total Reward')

% Plot total steps of each episode
figure(2)
plot(jList)
xlabel('Episodes')
ylabel('Total Steps')

% Plot Value table
figure(3)
surf(0:numCols-1,0:numRows-1,QtableOpt)
title('Learned Value Table')
xlim([0,numCols-1])
ylim([0,numRows-1])
view(0,90)
        
        

        

            
            
            
    




