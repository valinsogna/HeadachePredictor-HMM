#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 16:12:50 2022

@author: cyberguli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:15:51 2022

@author: cyberguli
"""
import matplotlib.pyplot as plt

import pyro.contrib.autoguide as autoguide
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pyro import poutine
import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from statsmodels.tsa.stattools import bds
import pandas as pd
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pyro.infer import Predictive

import numpy as np
figsize=(10,4)
pyro.clear_param_store()


series=read_csv('datipml.csv', header=0,names=["Data"])
plot_acf(series, title="original")
pyplot.savefig("original.png",dpi=900)

pyro.set_rng_seed(100)
x=read_csv('datipml.csv', header=0,names=["Data"])

temp=x.Data.tolist()

data=[]
for i in range(len(temp)):
    data.append(torch.tensor(temp[i]*1.))

datatrain=data[0:245]
datatest=data[245:]


'''
def modelA(data,start):
    z=[]
    k=pyro.param("k", torch.tensor(2.0))
    alpha1=pyro.param("alpha1",torch.tensor([1.,1.,1.,1.,1.,1.]),constraint=constraints.positive)
    beta1=pyro.param("beta1",torch.tensor([1.,1.,1.,1.,1.,1.]),constraint=constraints.positive)
    alpha2=pyro.param("alpha2",torch.tensor(1.),constraint=constraints.positive)
    beta2=pyro.param("beta2",torch.tensor(1.),constraint=constraints.positive)
    for i in range(len(data)):
        if i==0:
            z.append(pyro.sample("x_{}".format(i),dist.Categorical(torch.ones(6)/6)))
        else:
            l=pyro.sample("l_{}".format(i),dist.Beta(alpha2,beta2))
            phi=pyro.sample("phi_{}".format(i),dist.Dirichlet(alpha1))
            theta=[pyro.sample("theta_{}_{}".format(i,j),dist.Dirichlet(beta1)) for j in range(6)]
            index=pyro.sample("index_{}".format(i),dist.Categorical(phi))
            z.append(l*pyro.sample("x_{}".format(i),dist.Categorical(theta[index]))+(1-l)*z[i-1])
            pyro.sample("obs_{}".format(i), dist.Bernoulli(torch.exp(-z[i]**2/(2*k))), obs=data[i])





def guideA(data,start):
    q=pyro.param("q",torch.ones(6)/6, constraint=constraints.simplex)
    # register the two variational parameters with Pyro.
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    for i in range(len(data)):
        pyro.sample("x_{}".format(i), dist.Categorical(q))
'''

def modelA(data,start):
    z=[0]
    k=pyro.param("k", torch.tensor(2.0))
    alpha1=pyro.param("alpha1",torch.tensor([1.,1.,1.,1.,1.,1.]),constraint=constraints.positive)
    beta1=pyro.param("beta1",torch.tensor([1.,1.,1.,1.,1.,1.]),constraint=constraints.positive)
    alpha2=pyro.param("alpha2",torch.tensor(1.),constraint=constraints.positive)
    beta2=pyro.param("beta2",torch.tensor(1.),constraint=constraints.positive)
    for i in range(len(data)):
        l=pyro.sample("l_{}".format(i),dist.Beta(alpha2,beta2))
        phi=pyro.sample("phi_{}".format(i),dist.Dirichlet(alpha1))
        theta=[pyro.sample("theta_{}_{}".format(i,j),dist.Dirichlet(beta1)) for j in range(6)]
        index=pyro.sample("index_{}".format(i),dist.Categorical(phi))
        z.append(l*pyro.sample("x_{}".format(i),dist.Categorical(theta[index]))+(1-l)*z[i-1])
        pyro.sample("obs_{}".format(i), dist.Bernoulli(torch.exp(-z[i]**2/(2*k))), obs=data[i])



guideA = autoguide.AutoLaplaceApproximation(poutine.block(modelA, hide=['index_{}'.format(i) for i in range(len(data))]+['x_{}'.format(i) for i in range(len(data))]))
#guideA= autoguide.AutoStructured(modelA)





def BIC(data,start,k,alpha1,beta1,alpha2,beta2):
    z=[]

    #k=pyro.param(taorch.tensor(1.),constraint=constraints.positive)
    s=0
    N=len(data)
    for i in range(len(data)):
        if i==0:
            z.append(pyro.sample("x_{}".format(i),dist.Normal(start,1.0)))
            s=s+pyro.sample("obs_{}".format(i), dist.Bernoulli(torch.exp(-z[i]**2/(2*k))))
        else:
            l=pyro.sample("l_{}".format(i),dist.Beta(alpha2,beta2))
            phi=pyro.sample("phi_{}".format(i),dist.Dirichlet(alpha1))
            theta=[pyro.sample("theta_{}_{}".format(i,j),dist.Dirichlet(beta1)) for j in range(6)]
            index=pyro.sample("index_{}".format(i),dist.Categorical(phi))
            z.append(l*pyro.sample("x_{}".format(i),dist.Categorical(theta[index]))+(1-l)*z[i-1])
            s=s+dist.Bernoulli(torch.exp(-z[i]**2/(2*k))).log_prob(data[i])

    return 12*torch.log(torch.tensor(N))-2*s




# set up the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(modelA, guideA, optimizer, loss=Trace_ELBO())


n_steps = 5000
loss=[]
x
tmp=svi.evaluate_loss(data,0)
# do gradient steps
losses=[]
for step in range(n_steps):
    tmpold=tmp
    loss=svi.step(data,0)
    losses.append(loss)
    if step % 100 == 0:
        print('.', end='')
fig, ax = plt.subplots(figsize=figsize)
ax.plot(losses)
ax.set_title("Mixture Categorical loss");
fig.savefig("mixturecategorical.png",dpi=900)


alpha1=pyro.param("alpha1")
beta1=pyro.param("beta1")
alpha2=pyro.param("alpha2")
beta2=pyro.param("beta2")
k = pyro.param("k").item()



pred=[]
zeta=[]


    #k=pyro.param(torch.tensor(1.),constraint=constraints.positive)
k=pyro.param("k", torch.tensor(2.0))
alpha=pyro.param("alpha",torch.tensor([1.,1.,1.,1.,1.,1.]),constraint=constraints.positive)
beta=pyro.param("beta",torch.tensor([1.,1.,1.,1.,1.,1.]),constraint=constraints.positive)
    
#k=pyro.param(taorch.tensor(1.),constraint=constraints.positive)
s=0
N=len(data)
for i in range(len(data)):
    if i==0:
        zeta.append(pyro.sample("x_{}".format(i),dist.Normal(0,1.0)))
        pred.append(pyro.sample("obs_{}".format(i), dist.Bernoulli(torch.exp(-zeta[i]**2/(2*k)))))
    else:
        l=pyro.sample("l_{}".format(i),dist.Beta(alpha2,beta2))
        phi=pyro.sample("phi_{}".format(i),dist.Dirichlet(alpha1))
        theta=[pyro.sample("theta_{}_{}".format(i,j),dist.Dirichlet(beta1)) for j in range(6)]
        index=pyro.sample("index_{}".format(i),dist.Categorical(phi))
        zeta.append(l*pyro.sample("x_{}".format(i),dist.Categorical(theta[index]))+(1-l)*zeta[i-1])
        pred.append(pyro.sample("obs_{}".format(i), dist.Bernoulli(torch.exp(-zeta[i]**2/(2*k)))))






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


bic=BIC(datatest,0,k,alpha1,beta1,alpha2,beta2)

datatest=datatest.numpy()
print("BIC is" ,bic)


print("Autocorrelation STD is ", np.linalg.norm(autocorr(datatest)-autocorr(predtest))/len(datatest))

