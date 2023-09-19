#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:15:51 2022

@author: cyberguli
"""


from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from statsmodels.tsa.stattools import bds
import pandas as pd
from pyro import poutine
from pyro.optim import Adam
import pyro.contrib.autoguide as autoguide
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pyro.infer import Predictive

import numpy as np
figsize=(10,4)
pyro.clear_param_store()


series=read_csv('datipml.csv', header=0,names=["Data"])
plot_acf(series, title="ACF of the data")
pyplot.savefig("original.png",dpi=900)

pyro.set_rng_seed(100)
x=read_csv('datipml.csv', header=0,names=["Data"])

temp=x.Data.tolist()

data=[]
for i in range(len(temp)):
    data.append(torch.tensor(temp[i]*1.))

datatrain=data[0:245]
datatest=data[245:]

def modelA(data,start):
    z=[]
    k=pyro.param("k", torch.tensor(2.0))
    alpha1=pyro.param("alpha1",torch.tensor(2.),constraint=constraints.positive)
    beta1=pyro.param("beta1",torch.tensor(1.),constraint=constraints.positive)
    mmu=pyro.param("mmu",torch.tensor(0.5))
    smu=pyro.param("smu",torch.tensor(2),constraint=constraints.positive)
    alpha2=pyro.param("alpha2",torch.tensor(1.),constraint=constraints.positive)
    beta2=pyro.param("beta2",torch.tensor(1.),constraint=constraints.positive)
    phi=pyro.param("phi",torch.ones(6)/6, constraint=constraints.simplex)
    
    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
    for i in range(len(data)):
        if i==0:
            z.append(pyro.sample("normal_{}".format(i),dist.Normal(start,1.0)))
        else:
            l=pyro.sample("l_{}".format(i),dist.Beta(alpha1,beta1))
            mu=[pyro.sample("mu_{}_{}".format(i,j),dist.Normal(mmu,smu)) for j in range(6)]
            sigma=[pyro.sample("sigma_{}_{}".format(i,j),dist.InverseGamma(alpha2,beta2)) for j in range(6)]
            index=pyro.sample("index_{}".format(i),dist.Categorical(phi))
            z.append(l*z[i-1]+(1-l)*pyro.sample("normal_{}".format(i),dist.Normal(mu[index],sigma[index])))
        pyro.sample("obs_{}".format(i), dist.Bernoulli(torch.exp(-z[i]**2/(2*k))), obs=data[i])



guideA = autoguide.AutoNormal(poutine.block(modelA, hide=['index_{}'.format(i) for i in range(len(data))]))
'''
def guideA(data,start):
    # register the two variational parameters with Pyro.
    alpha_q = pyro.param("alpha_q", torch.tensor(0.0))
    beta_q = pyro.param("beta_q", torch.tensor(1.0))
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    for i in range(len(data)):
        pyro.sample("normal_{}".format(i), dist.Normal(alpha_q, beta_q))
'''


def BIC(data,start,k,alpha1,beta1,mmu,smu,alpha2,beta2,phi):
    z=[]
    s=0
    N=len(data)
    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
    for i in range(len(data)):
        if i==0:
            z.append(pyro.sample("normal_{}".format(i),dist.Normal(start,1.0)))
        else:
            l=pyro.sample("l_{}".format(i),dist.Beta(alpha1,beta1))
            mu=[pyro.sample("mu_{}_{}".format(i,j),dist.Normal(mmu,smu)) for j in range(6)]
            sigma=[pyro.sample("sigma_{}_{}".format(i,j),dist.InverseGamma(alpha2,beta2)) for j in range(6)]
            index=pyro.sample("index_{}".format(i),dist.Categorical(phi))
            z.append(l*z[i-1]+(1-l)*pyro.sample("normal_{}".format(i),dist.Normal(mu[index],sigma[index])))
        s=s+dist.Bernoulli(torch.exp(-z[i]**2/(2*k))).log_prob(data[i])

    return 12*torch.log(torch.tensor(N))-2*s


# set up the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(modelA, guideA, optimizer, loss=Trace_ELBO())


n_steps = 5000
tmp=svi.evaluate_loss(data,0)
losses=[]
# do gradient steps
for step in range(n_steps):
    tmpold=tmp
    loss=svi.step(data,0)
    tmp=svi.evaluate_loss(data,0)
    losses.append(loss)
    if step % 100 == 0:
        print('.', end='')

fig, ax = plt.subplots(figsize=figsize)
ax.plot(losses)
ax.set_title("Mixture Gaussian loss");
fig.savefig("mixturegaussian.png",dpi=900)


alpha1=pyro.param("alpha1")
alpha2=pyro.param("alpha2")
beta1=pyro.param("beta1")
beta2=pyro.param("beta2")
phi=pyro.param("phi")
mmu=pyro.param("mmu")
smu=pyro.param("smu")


k = pyro.param("k").item()



pred=[]
zeta=[]


    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
#for i in range(len(data)):
with pyro.plate("data", len(data)):
    if i==0:
        zeta.append(pyro.sample("normal_{}".format(i),dist.Normal(0,1.0)))
        pred.append(pyro.sample("obs_{}".format(i), dist.Bernoulli(torch.exp(-zeta[i]**2/(2*k)))))
        
    else:
        l=pyro.sample("l_{}".format(i),dist.Beta(alpha1,beta1))
        mu=[pyro.sample("mu_{}_{}".format(i,j),dist.Normal(mmu,smu)) for j in range(6)]
        sigma=[pyro.sample("sigma_{}_{}".format(i,j),dist.InverseGamma(alpha2,beta2)) for j in range(6)]
        index=pyro.sample("index_{}".format(i),dist.Categorical(phi))
        zeta.append(l*zeta[i-1]+(1-l)*pyro.sample("normal_{}".format(i),dist.Normal(mu[index],sigma[index])))
        pred.append(pyro.sample("obs_{}".format(i), dist.Bernoulli(torch.exp(-zeta[i]**2/(2*k)))))





def compute_error(list1,list2):
    err=0
    for i in range(len(list1)):
        if abs(list1[i]-list2[i])>10e-02:
            err=err+1
    
    err=err/len(list1)
    return err
pred=torch.tensor(pred).reshape(len(pred)).numpy()
predtest=pred[245:]
predtrain=pred[:245]
datatest=torch.tensor(datatest).reshape(len(datatest))

def autocorr(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    if variance==0:
        return 1
    else:
        result = r/(variance*(np.arange(n, 0, -1)))
    return result


bic=BIC(datatest,0,k,alpha1,beta1,mmu,smu,alpha2,beta2,phi)

datatest=datatest.numpy()
print("BIC is" ,bic)


print("Autocorrelation STD is ", np.linalg.norm(autocorr(datatest)-autocorr(predtest))/len(datatest))

plot_acf(pd.DataFrame(torch.tensor(data)-torch.tensor(pred)), title="Weiner unit residuals")
pyplot.savefig("weiner.png",dpi=900)

