import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.tsa.stattools import bds
import pandas as pd
from pyro.infer import Predictive
import pyro.contrib.autoguide as autoguide
from pyro.infer.mcmc import MCMC, HMC, NUTS
import warnings

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000

assert pyro.__version__.startswith('1.8.1')

# clear the param store in case we're in a REPL
pyro.clear_param_store()



series=read_csv('datipml.csv', header=None,names=["Data"])
plot_acf(series, title="original")
pyro.set_rng_seed(100)
x=read_csv('datipml.csv', header=None,names=["Data"])

temp=x.Data.tolist()
k=6
data=[]
for i in range(len(temp)):
    data.append(torch.tensor(temp[i]*1.))

datatrain=data[0:245]
datatest=data[245:]


def model(data,k):
    if k==6:
        A=pyro.param("A",torch.tensor([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,0,0,0,0,0]],dtype=torch.float64),constraint=constraints.simplex)
# Emission Probabilities
        B=pyro.param("B", torch.tensor([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]],dtype=torch.float64),constraint=constraints.simplex)
        x0=torch.tensor([1,0,0,0,0,0],dtype=torch.float64)
    else:
        A=pyro.param("A",torch.ones(k,k)/k)
        B=pyro.param("B",torch.ones(k,2)/2)
        x0=torch.ones(k)/k
    N=len(data)
    x=[]
    for i in range(N):
        x.append(pyro.sample('obs_{}'.format(i),dist.Categorical(torch.matmul(x0,B)),obs=data[i]))
        x0=torch.matmul(x0,A)
    
    return x

def modelpred(data,A,B):
    k=A.size()[0]
    if k==6:
        x0=torch.tensor([1,0,0,0,0,0],dtype=torch.float64)
    else:
        x0=torch.ones(k)/k
    N=len(data)
    x=[]
    for i in range(N):
        x.append(pyro.sample('obs_{}'.format(i),dist.Categorical(torch.matmul(x0,B))))
        x0=torch.matmul(x0,A)
    
    return x


kernel=HMC(model)
mcmc = MCMC(kernel,num_samples=200,warmup_steps=100, num_chains=3)
posterior=mcmc.run(data=datatrain,k=6)
        
A=pyro.param("A")
B=pyro.param("B")        
        
y=modelpred(data,A,B)
ytest=y[245:]
yhat_train=y[:245]
resid=(torch.tensor(yhat_train)-torch.tensor(datatrain)).numpy()
print("BDS residual:",torch.mean(torch.tensor((bds(resid,12))[1])))
print("Test error:",torch.mean(abs(torch.tensor(datatest)-torch.tensor(ytest))))
ytest=y[245:]
plot_acf(pd.DataFrame(torch.tensor(data)-torch.tensor(y)), title="Weiner unit residuals")
