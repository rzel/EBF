%// This make.m is for MATLAB
%// Function: compile c++ files which rely on OpenCV for Matlab using mex
%// Author : zouxy
%// Date   : 2014-03-05
%// HomePage : http://blog.csdn.net/zouxy09
%// Email  : zouxy09@qq.com

%% Please modify your path of OpenCV
%% If your have any question, please contact Zou Xiaoyi

% Notice: first use "mex -setup" to choose your c/c++ compiler

%-------------------------------------------------------------------
%% get the architecture of this computer
is_64bit = strcmp(computer,'MACI64') || strcmp(computer,'GLNXA64') || strcmp(computer,'PCWIN64');


%-------------------------------------------------------------------
%% the configuration of compiler
% You need to modify this configuration according to your own path of OpenCV
% Notice: if your system is 64bit, your OpenCV must be 64bit!
out_dir='./';
CPPFLAGS = ' -O -DNDEBUG -I.\demo_ASIFT_src\';  % your OpenCV "include" path
LDFLAGS = ' -L.\lib\ ';	% your OpenCV "lib" path
LIBS = ' -ldemo_ASIFT';
if is_64bit
	CPPFLAGS = [CPPFLAGS ' -largeArrayDims'];
end
%% add your files here!
compile_files = { 
	% the list of your code files which need to be compiled
	'mexASIFT.cpp',
  
};


%-------------------------------------------------------------------
%% compiling...
for k = 1 : length(compile_files)
    str = compile_files{k};
    fprintf('compilation of: %s\n', str);
    str = [str ' -outdir ' out_dir CPPFLAGS LDFLAGS LIBS];
    args = regexp(str, '\s+', 'split');
    mex(args{:});
end

fprintf('Congratulations, compilation successful!!!\n');