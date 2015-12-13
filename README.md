# mompy

mompy is a package for solving polynomial optimization and the Generalized Moment Problem. 
Examples, background and instructions can be found in these ipython worksheets:

[polynomial optimization worksheet](https://github.com/sidaw/mompy/blob/master/polynomial_optimization.ipynb)
[extra examples worksheet](https://github.com/sidaw/mompy/blob/master/extra_examples.ipynb)

This was implemented for our paper

Sida I. Wang, Arun Chaganty, Percy Liang, [Estimating mixture models via mixtures of polynomials](http://papers.nips.cc/paper/5702-estimating-mixture-models-via-mixtures-of-polynomials.pdf), NIPS 2015


## Polynomial Optimization

The polynomial optimization problem is to
```
minimize f(x) subject to g_i(x) >= 0, for i=1,...,N.
```
where x is a real vector and f(x), g_i(x) are polynomials.
mompy is an implementation of the techniques developed by Parrilo et al. and Lasserre et al. 
where this problem is relaxed to a semidefinite program.

To minimize f subject to constraints g, one can just write:

```python
x,y = sp.symbols('x,y')
f =  x**2 + y**2
gs = [x+y>=4, x+y<=4]
sol = mp.solvers.solve_GMP(f, gs)
```

[GloptiPoly paper](http://homepages.laas.fr/henrion/papers/gloptipoly3.pdf)
[solution extraction paper](http://homepages.laas.fr/henrion/papers/extract.pdf) are reproduced.

In the second part, we give the simplest example of estimating a mixture model by solving the Generalized Moment Problem (GMP). 
Some extensions can be found in [the extra examples worksheet](extra_examples.ipynb). 
This software was used for our paper
where we look at some more elaborate settings for the mixture model problem.

Other implementation for solving the polynomial optimization problem / and Generalized Moment Problem are
[GloptiPoly](http://homepages.laas.fr/henrion/software/gloptipoly/) and 
[SOSTOOLS](http://www.cds.caltech.edu/sostools/) in MATLAB, and
[ncpol2sdpa](https://pypi.python.org/pypi/ncpol2sdpa/) in python described [here](http://arxiv.org/abs/1308.6029).
