import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

##Possion
x = np.arange(0, 20, 1)
mu=0.5

#true distribution
distribution_true=stats.poisson.pmf(x,1/mu)
plt.plot(x,distribution_true, '-o',label=mu)
plt.legend()
plt.show()


#simulation
sample={}
fig_sample=plt.figure(figsize=(12,8))
for i,n in enumerate([20,200,2000,20000]):
    ax = fig_sample.add_subplot(2, 2, i+1)
    sample[i] = stats.poisson.rvs(1/mu,size=n)
    ax.title.set_text(f'sample size: {n}')
    ax.hist(sample[i])
plt.show()


#MLE
fig_MLE=plt.figure(figsize=(12,8))
for i in range(0,len(sample)):
    estimator_MLE=sample[i].mean()

    ax = fig_MLE.add_subplot(2, 2, i+1)
    ax.title.set_text(f'sample size: {len(sample[i])}')
    ax.plot(distribution_true,label='Actual')
    ax.hist(sample[i],density=True,label='sample')
    ax.axvline(estimator_MLE, 0, 1, color='r', label='MLE estimator(sample mean)')
    ax.axvline(1/mu, 0, 1, color='g', label='Actual mu')
    ax.legend()
plt.show()



#MAP
alpha_=1
lambda_=1
prior=stats.gamma(a=alpha_,scale=1/lambda_)
plt.plot(prior.pdf(x))

fig_MAP=plt.figure(figsize=(12,8))
for i in range(0,len(sample)):
    posterior=stats.gamma(a=alpha_+np.sum(sample[i]), scale=1/(lambda_+len(sample[i])))
    distribution_MAP=posterior.pdf(x)
    estimator_MAP=posterior.moment(1)

    ax = fig_MAP.add_subplot(2, 2, i+1)
    ax.title.set_text(f'sample size: {len(sample[i])}')
    ax.plot(distribution_MAP,label='MAP distribution')
    ax.axvline(estimator_MAP, 0, 1, color='r', label='MAP estimator(sample mean)')
    ax.axvline(1/mu, 0, 1, color='g', label='Actual mu')
    ax.legend()
plt.show()

