function [img_addnoise] = noise(img_gray)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
sigma=20;
img_addnoise=double(img_gray)+sigma*randn(size(img_gray));
end

