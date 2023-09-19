import math
import os
import torch
import torch.distributions.constraints as constraints
import matplotlib.pyplot as plt
from matplotlib import pyplot
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
import warnings 
import numpy as np
warnings.filterwarnings("ignore")
figsize=(10,4)

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
        '''
        A=pyro.param("A",torch.tensor([[0.02,0.90,0.02,0.02,0.02,0.02],[0.02,0.02,0.90,0.02,0.02,0.02],[0.02,0.02,0.02,0.90,0.02,0.02],[0.02,0.02,0.02,0.02,0.9,0.02],[0.02,0.90,0.02,0.02,0.02,0.02],[0.90,0.02,0.02,0.02,0.02,0.02]],dtype=torch.float64),constraint=constraints.simplex)
# Emission Probabilities
        B=pyro.param("B", torch.tensor([[0.9,0.1],[0.9,0.1],[0.9,0.1],[0.1,0.9],[0.1,0.9],[0.1,0.9]],dtype=torch.float64),constraint=constraints.simplex)
        '''
        A=pyro.param("A",torch.tensor([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,0,0,0,0,0]],dtype=torch.float64),constraint=constraints.simplex)
# Emission Probabilities
        B=pyro.param("B", torch.tensor([[0.95,0.05],[0.88,0.12],[0.88,0.12],[0.12,0.88],[0.12,0.88],[0.12,0.88]],dtype=torch.float64),constraint=constraints.simplex)      
        x0=pyro.param("x0",torch.tensor([1,0,0,0,0,0],dtype=torch.float64),constraint=constraints.simplex)
    else:
        A=pyro.param("A",dist.Dirichlet(torch.ones(k,k)/k).sample(),constraint=constraints.simplex)
        B=pyro.param("B",dist.Dirichlet(torch.ones(k,2)/2).sample(),constraint=constraints.simplex)
        x0=pyro.param("x0",torch.ones(k)/k,constraint=constraints.simplex)
    N=len(data)
    x=[]
    for i in range(N):
        x.append(pyro.sample('obs_{}'.format(i),dist.Categorical(torch.matmul(x0,B)),obs=data[i]))
        x0=torch.matmul(x0,A)
    
    return x

def modelpred(data,A,B,x0):
    N=len(data)
    x=[]
    for i in range(N):
        x.append(pyro.sample('obs_{}'.format(i),dist.Categorical(torch.matmul(x0,B))))
        x0=torch.matmul(x0,A)
    
    return x


def BIC(data,A,B,x0):
    N=len(data)
    x=[]
    s=0
    for i in range(N):
        s=s+dist.Categorical(torch.matmul(x0,B)).log_prob(data[i])
        x0=torch.matmul(x0,A)
    
    return (A.shape[0]**2)*torch.log(torch.tensor(N))-s


guide = autoguide.AutoDiscreteParallel(model)

adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
n_steps = 5000
losses=[]
# do gradient steps
for step in range(n_steps):
    loss=svi.step(data,k)
    losses.append(loss)
    if step % 100 == 0:
        print('.', end='')

fig, ax = plt.subplots(figsize=figsize)
ax.plot(losses)
ax.set_title("DTMC loss");
fig.savefig("dtmcloss.png",dpi=900)
        
A=pyro.param("A")
B=pyro.param("B") 
x0=pyro.param("x0")       
        
y=modelpred(data,A,B,x0)
pred=modelpred(data,A,B,x0)
pred=torch.tensor(pred).reshape(len(pred)).numpy()
predtest=pred[245:]
predtrain=pred[:245]
print("BIC is", BIC(datatrain,A,B,x0))
datatest=torch.tensor(datatest).reshape(len(datatest)).numpy()
datatrain=torch.tensor(datatrain).reshape(len(datatrain)).numpy()

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

print("Autocorrelation STD is ", np.linalg.norm(autocorr(datatrain)-autocorr(predtrain))/len(datatrain))


yhat_test=y[245:]
yhat_train=y[:245]
resid=(torch.tensor(yhat_train)-torch.tensor(datatrain)).numpy()
#print("Test error:",torch.mean(abs(torch.tensor(datatest)-torch.tensor(ytest))))
ytest=y[245:]
plot_acf(pd.DataFrame(torch.tensor(data)-torch.tensor(y)), title="Weiner unit residuals")
pyplot.savefig("dtmc.png",dpi=900)

'''
resid_train=(torch.tensor(datatrain)-torch.tensor(yhat_train)).numpy()
resid_test=(torch.tensor(datatest)-torch.tensor(yhat_test)).numpy()
yhat_train=torch.tensor(yhat_train).reshape(245).numpy().reshape(-1,1)
resid_train=torch.tensor(resid_train).reshape(245).numpy()
yhat_test=torch.tensor(yhat_test).reshape(85).numpy().reshape(-1,1)
resid_test=torch.tensor(resid_test).reshape(85).numpy()
gnb = MultinomialNB()
y_pred = gnb.fit(yhat_train, resid_train).predict(yhat_test)
'''

'''
resid_train=(torch.tensor(datatrain)-torch.tensor(yhat_train))
resid_test=(torch.tensor(datatest)-torch.tensor(yhat_test))
yhat_train=torch.tensor(yhat_train).reshape(245).to(torch.int64)
resid_train=torch.tensor(resid_train).reshape(245)
yhat_test=torch.tensor(yhat_test).reshape(85).to(torch.int64)
resid_test=torch.tensor(resid_test).reshape(85)

temp=torch.zeros(2,3)
for i in range(len(yhat_train)):
    for k in range(2):
        for h in range(-1,2):
            if yhat_train[i]==k and resid_train[i]==h:
                    temp[k,h+1]=temp[k,h+1]+1

for k in range(2):
    temp[k,]=temp[k,]/torch.sum(temp[k,])
    
print(temp)

def model2(y,x):
    resid=torch.zeros(len(x))
    p=pyro.param("p", torch.tensor([[0.0000, 0.8862, 0.1138],[0.1230, 0.8770, 0.0000]]), constraint=constraints.simplex)    
    for i in range(len(x)):
        resid[i]=pyro.sample("resid_{}".format(i), dist.Categorical(p[x[i]]),obs=y[i]+1)

    return resid

def modelpred2(x):
    resid=torch.zeros(len(x))
    p=pyro.param("p", torch.tensor([[0.0000, 0.8862, 0.1138],[0.1230, 0.8770, 0.0000]]), constraint=constraints.simplex)    
    for i in range(len(x)):
        resid[i]=pyro.sample("resid_{}".format(i), dist.Categorical(p[x[i]]))

    return resid

guide2 = autoguide.AutoDelta(model2)

adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model2, guide, optimizer, loss=Trace_ELBO())
n_steps = 500
# do gradient steps
for step in range(n_steps):
    svi.step(resid_train,yhat_train)
    if step % 100 == 0:
        print('.', end='')

resid_pred=modelpred2(yhat_train)
'''

