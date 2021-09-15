#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from scipy import linalg
import numpy as np

"""
param: t 解密需要的最少碎片数
       n 生成的秘钥碎片数
       p 取模所用的质数
"""
M = 28
T = 3
N = 4
P = 29


# In[2]:


"""
Shamir算法的明文加密函数
param： m 明文字符
        t 多项式的个数，最高次为t-1
        n 生成的秘钥碎片个数
        p 取模所用的质数
return： 采用的参数，x,y组成的秘钥碎片对
"""
def shamirEncode(m, t, n, p):
    parameter = [random.randint(1,100) for i in range(t-1)]
    parameter = [m] + parameter
    x = [random.randint(1,10) for i in range(n)]
    #ensure x are different with each other
    while(len(set(x))<4):
        x = [random.randint(1,50) for i in range(n)]
    y = [getY(temp_x, parameter)%p for temp_x in x]
    return parameter, x, y


# In[3]:


"""
计算X值对应Y的函数
param： x x的值
        parameter 多项式的t个参数
return： 对应的Y的值
"""
def getY(x, parameter):
    length = len(parameter)
    y = 0
    for i in range(0,length):
        y += parameter[i]*(x**i)
    return y


# In[4]:


"""
Shamir算法的密文解密函数
param： x x数组包含多个秘钥碎片
        y y数组包含多个秘钥碎片
        t 最少需要的秘钥个数
        p 公开的所模的数即全局变量P
return： result 解密后的明文
"""
def shamirDecode(x,y,t,p):
    if(len(x)<t or len(y)<t):
        print("error：缺少信息，无法解密！")
        return -1
    b = []
    j = 0
    result = 0
    #x1*x2*x3...
    x_multi = 1  
    for i in range(0,t):
        x_multi *= x[i]
        
    while(j<t):
#         print("-----------------------------", j,"--------------------------")
        b_temp = x_multi / x[j]
        for i in range(0,t):
            
            if i == j:
                continue
            else:
                numNeedInv = (x[i]-x[j])%p
                _,inv,_ = ext_gcd(p,numNeedInv)
                inv = inv%p
                b_temp *=  inv
#                 print(inv)
#         print(b_temp)
        b.append(b_temp%p)
        j += 1
#     print(x,y,b)
    for i in range(0,t):
        result += b[i]*y[i]
#         print(result)
    return result%p


# In[5]:


"""
求逆元的函数
param： a 所模的数
        b 求逆元的数 bx = 1 mod a
return： x为与a的乘数 gcd(a,b) = ax + by
         y为b的逆元
         gcd为最大公因子
"""
def ext_gcd(a, b): #扩展欧几里得算法    
    if b == 0:          
        return 1, 0, a     
    else:         
        x, y, gcd = ext_gcd(b, a % b) #递归直至余数等于0(需多递归一层用来判断)        
        x, y = y, (x - (a // b) * y) #辗转相除法反向推导每层a、b的因子使得gcd(a,b)=ax+by成立      
        return x, y, gcd


# In[6]:


param ,x , y = shamirEncode(M,T, N, P)


# In[7]:


shamirDecode(x,y,T,P)

