# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 14:50:46 2025

@author: CUUUI
"""
import torch

x = torch.arange(4.0)
x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print('y=', y)

y.backward()
# print(y.backward())
print(x.grad)

# 默认情况下，pytorch会累积梯度，因此需要清除之前的值
x.grad.zero_()
y = x.sum()
print(y)
y.backward()
print(x.grad)

