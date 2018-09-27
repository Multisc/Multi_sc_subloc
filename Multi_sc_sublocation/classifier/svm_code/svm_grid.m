function svm_grid
tic;
close all;
clear;
clc; 
format compact;

% CoreNum=48; %调用的处理器个数
% if parpool('local')<=0  %之前没有打开
%     parpool('open','local',CoreNum);
% else  %之前已经打开
%     disp('matlab pool already started');
% end

data=xlsread('F:\98_data_feature.xlsx');
large=96;   %标签数
labels=xlsread('F:\label.xlsx','A1:A96');
temp=[]; 
cg=[]; 
train_98=data(2:large,:);
train_98_labels =labels(2:large); 
test_98 =data(1,:); 
test_98_labels =labels(1); 
%% 选择GA最佳的SVM参数c&g   第一次
[bestaac,bestc,bestg] = SVMcgForClass(train_98_labels, train_98,-10,10,-10,10); %返回参数必须是一个或者三个
%网格参数寻优函数(分类问题)[bestCVaccuracy,bestc,bestg]=SVMcgForClass(train_label,train,
%cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)罚参数c的变化范围为-10到10
%RBF核参数g的变化范围为-10到10
% cg(1,1)=bestc; %最好的c值给第一行第一列
% cg(1,2)=bestg; %同上
%%%%
%% 利用最佳的参数进行SVM网络训练  第一次
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
%%%%%
model = svmtrain(train_98_labels, train_98,cmd); %model是训练得到的模型
[accuracy] = svmpredict(test_98_labels, test_98, model); 
temp(1)=accuracy(1);  %把第一个精确度矩阵赋值给temp
temp_a=(1:large-2);
parfor n=2:(large-1) 
train_98 = [data(1:(n-1),:);data((n+1):large,:)];% 
train_98_labels = [labels(1:(n-1));labels((n+1):large)];%
test_98 =data(n,:); 
test_98_labels =labels(n); 
%% 选择GA最佳的SVM参数c&g  第二次
[bestaac,bestc,bestg] = SVMcgForClass(train_98_labels, train_98,-10,10,-10,10);
% cg(n,1)=bestc; %最好的C值变给第n行1列  n从2到96
% cg(n,2)=bestg;%最好的g值变给n行2列  
%%%%
%% 利用最佳的参数进行SVM网络训练  第二次
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
%%%%%
model = svmtrain(train_98_labels, train_98,cmd);
[accuracy] = svmpredict(test_98_labels, test_98, model);
% temp_a = [temp_a, accuracy(1)];
temp_a(n-1)=accuracy(1);
end
train_98=data(1:(large-1),:);
train_98_labels =labels(1:(large-1));
test_98 =data(large,:); 
test_98_labels =labels(large);
%% 选择GA最佳的SVM参数c&g   第三次
[bestaac,bestc,bestg] = SVMcgForClass(train_98_labels, train_98,-10,10,-10,10);
% cg(large,1)=bestc;  %把c赋值给96行第一列
% cg(large,2)=bestg; %同上
%%%%
%% 利用最佳的参数进行SVM网络训练   第三次
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
%%%%%
model = svmtrain(train_98_labels, train_98,cmd);
[accuracy] = svmpredict(test_98_labels, test_98, model);
temp(large)=accuracy(1);  %把accuracy第一个元素赋值给temp的第96个元素
temp(2:large-1)=temp_a;
label = reshape(labels , 1 , large );
result = (temp==label);
wrong_num=large-(length(nonzeros(result)));
real_acc=(length(nonzeros(result)))/large;
fprintf(' _num= %d \n',wrong_num);
fprintf(' real_acc= %f \n',real_acc);
for m=1:large
xlswrite('F:\label.xlsx',temp(m),'sheet1',['B',num2str(m)]); %写入这个文件A列 数字变为字符串 
end
wrong=fopen('wrong_mean.txt','a');%'A.txt'为文件名；'a'为打开方式：在打开的文件末端添加数据，若文件不存在则创建。
accr=fopen('acc_mean.txt','a');%'A.txt'为文件名；'a'为打开方式：在打开的文件末端添加数据，若文件不存在则创建。
fprintf(wrong,'%d \r\n',wrong_num);%fp为文件句柄，指定要写入数据的文件。注意：%d后有空格。
fprintf(accr,'%d \r\n',real_acc);%fp为文件句柄，指定要写入数据的文件。注意：%d后有空格。
fclose(wrong);%关闭文件。
fclose(accr);%关闭文件。
toc;
%disp(cg);
function [bestacc,bestc,bestg] = SVMcgForClass(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%网格参数寻优函数
%对比上面函数 
%[bestc,bestg] = SVMcgForClass(train_98_labels, train_98,-10,10,-10,10);
%网格参数寻优函数(分类问题)[bestCVaccuracy,bestc,bestg]=SVMcgForClass(train_label,train,
%cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)罚参数c的变化范围为-10到10
%RBF核参数g的变化范围为-10到10
%SVMcg cross validation by faruto

%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto BNU
%last modified 2010.01.17
%Super Moderator @ www.ilovematlab.cn

% 若转载请注明：
% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2009. 
% Software available at http://www.ilovematlab.cn
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

% about the parameters of SVMcg 
if nargin < 10
    accstep = 4.5;
end
if nargin < 8
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 7
    v = 5;
end
if nargin < 5
    gmax = 8;
    gmin = -8;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end
% X:c Y:g cg:CVaccuracy
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);

eps = 10^(-4);

% record acc with different c & g,and find the bestacc with the smallest c
bestc = 1;
bestg = 0.1;
bestacc = 0;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) )];
        cg(i,j) = svmtrain(train_label, train, cmd);
        
        if cg(i,j) <= 55
            continue;
        end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
        if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^X(i,j) 
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
    end
end
