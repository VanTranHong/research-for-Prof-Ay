import numpy as np
from scipy.stats.distributions import norm

## Simulation study to assess coverage probability of
## the confidence interval for the comparison of two
## population means.

## Number of simulated data sets
nrep = 1000

## Sample sizes for the two observed samples
nx,ny = 10,20

## Standard deviation for the sample from the second population
sigy = 2

## Estimate the coverage probability using simulation
CP = 0
for it in range(nrep):

    ## Generate a data set that satisfies the null hypothesis
    X = np.random.normal(size=nx)
    Y = sigy*np.random.normal(size=ny)

    ## The Z-score
    md = X.mean() - Y.mean()
    se = np.sqrt(X.var()/nx + Y.var()/ny)

    ## Check whether the interval contains the true value
    if (md - 2*se < 0) and (md + 2*se > 0):
        CP += 1

CP /= float(nrep)


## Vectorized version of the simulation study to assess the coverage
## probability of the confidence interval for the comparison of two
## population means.

## Generate nrep data sets simultaneously
X = np.random.normal(size=(nrep,nx))
Y = sigy*np.random.normal(size=(nrep,ny))

## Calculate Z-scores for all the data sets simultaneously
MD = X.mean(1) - Y.mean(1)
SE = np.sqrt(X.var(1)/nx + Y.var(1)/ny)

## The coverage probability
CP = np.mean( (MD - 2*SE < 0) & (MD + 2*SE > 0) )

## Modify the previous simulation study so that one of the 
## populations is exponential.

## Generate nrep data sets
X = -np.log(np.random.uniform(size=(nrep,nx)))
Y = 1 + sigy*np.random.normal(size=(nrep,ny))

## Calculate all the Z-scores
MD = X.mean(1) - Y.mean(1)
SE = np.sqrt(X.var(1)/nx + Y.var(1)/ny)

## The coverage probability
CP = np.mean( (MD - 2*SE < 0) & (MD + 2*SE > 0) )

## Simulation study for the log odds ratio

from scipy.stats import rv_discrete

## Cell probabilities
P = np.array([[0.3,0.2],[0.1,0.4]])

## The population log odds ratio
PLOR = np.log(P[0,0]) + np.log(P[1,1]) - np.log(P[0,1]) - np.log(P[1,0])

## Sample size
n = 100

## ravel vectorizes by row
m = rv_discrete(values=((0,1,2,3), P.ravel()))

## Generate the data
D = m.rvs(size=(nrep,n))

## Convert to cell counts
Q = np.zeros((nrep,4))
for j in range(4):
    Q[:,j] = (D == j).sum(1)

## Calculate the log odds ratio
LOR = np.log(Q[:,0]) + np.log(Q[:,3]) - np.log(Q[:,1]) - np.log(Q[:,2])

## The standard error
SE = np.sqrt((1/Q.astype(np.float64)).sum(1))

print "The mean estimated standard error is %.3f" % SE.mean()
print "The standard deviation of the estimates is %.3f" % LOR.std()

## 95% confidence intervals
LCL = LOR - 2*SE
UCL = LOR + 2*SE

## Coverage probability
CP = np.mean((PLOR > LCL) & (PLOR < UCL))

print "The population LOR is %.2f" % PLOR
print "The expected value of the estimated LOR is %.2f" % LOR[np.isfinite(LOR)].mean()
print "The coverage probability of the 95%% CI is %.3f" % CP

## Simulation study for the correlation coefficient

## Sample size
n = 20

## Correlation between X and Y
r = 0.3

## Generate matrices X and Y such that the i^th rows of X and Y are
## correlated with correlation coefficient 0.3.
X = np.random.normal(size=(nrep,n))
Y = r*X + np.sqrt(1-r**2)*np.random.normal(size=(nrep,n))

## Get all the correlation coefficients
R = [np.corrcoef(x,y)[0,1] for x,y in zip(X,Y)]
R = np.array(R)

## Fisher transform all the correlation coefficients
F = 0.5*np.log((1+R)/(1-R))

print "The standard deviation of the Fisher transformed " +\
      "correlation coefficients is %.3f" % F.std() 
print "1/sqrt(n-3)=%.3f" % (1/np.sqrt(n-3))

## 95% confidence intervals on the Fisher transform scale
LCL = F - 2/np.sqrt(n-3)
UCL = F + 2/np.sqrt(n-3)

## Convert the intervals back to the correlation scale
LCL = (np.exp(2*LCL)-1) / (np.exp(2*LCL)+1)
UCL = (np.exp(2*UCL)-1) / (np.exp(2*UCL)+1)

CP = np.mean((LCL < r) & (r < UCL))

print "The coverage probability is %.3f" % CP

## Simulation study for contingency tables

## Row and column cell probabilities
PR = np.r_[0.3, 0.4, 0.3]
PC = np.r_[0.6, 0.4]

## Cell probabilities satisfying row/column independence
T = np.outer(PR, PC)

## Modify the probabilities a bit to deviate from independence
x = 0.05 ## set to 0 to give independent rows and columns
T1 = T.copy()
T1[1,:] += [x, -x]

## Sample size
n = 100

## ravel vectorizes by row
m = rv_discrete(values=(range(6), T1.ravel()))

## Generate the data
D = m.rvs(size=(nrep,n))

## Convert to cell counts
Q = np.zeros((nrep,6))
for j in range(6):
    Q[:,j] = (D == j).sum(1)
Q = np.reshape(Q, (nrep, 3, 2))

## Row and column margins
RM = Q.sum(2) / float(n)
CM = Q.sum(1) / float(n)

## Fitted probabilities and counts under independence
E = [np.outer(rm,cm) for rm,cm in zip(RM,CM)]
C = [n*x for x in E]

## Chi-square statistics
CS = [((x-c)**2/c).sum() for x,c in zip(Q,C)]
CS = np.array(CS)

## Test statistic threshold for alpha=0.04
from scipy.stats.distributions import chi2
df = (3-1)*(2-1)
ts = chi2.ppf(0.95, df)

print "Proportion of chi^2 tests that reject the independence hypothesis: %.2f" %\
    np.mean(CS > ts)