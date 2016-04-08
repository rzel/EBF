clear; clc;
addpath(genpath('.'));

cd ./ASIFT
    compileASIFT;
cd ../


cd ./matching
    compileMatching;
cd ..

clear; 