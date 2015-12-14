# mompy

mompy is a package for solving polynomial optimization and the Generalized Moment Problem by relaxing to a semidefinite program. These techniques are described by Lasserre et al. and Parrilo el al. in the references.

Various examples are included in the
[polynomial optimization worksheet](https://github.com/sidaw/mompy/blob/master/polynomial_optimization.ipynb)
and the
[extra examples worksheet](https://github.com/sidaw/mompy/blob/master/extra_examples.ipynb), which includes a mixture of Gaussians example and a maxcut example. 

mompy was used in our paper, which also describes the estimating mixture model setup:

Sida I. Wang, Arun Tejasvi Chaganty, Percy Liang. [Estimating mixture models via mixtures of polynomials](http://papers.nips.cc/paper/5702-estimating-mixture-models-via-mixtures-of-polynomials.pdf). NIPS 2015

## Dependencies
You need [sympy](http://www.sympy.org/) and [cvxopt](http://cvxopt.org/) in order to use mompy.

## Problem setup

The polynomial optimization problem is to
```
minimize f(x) subject to g_i(x) >= 0, for i=1,...,N.
```
where x is a real vector and f(x), g_i(x) are polynomials.
mompy is an implementation of the Lassarre/SOS relaxations to a semidefinite program.
One can easily encode NP-hard problems in this so specialized solvers are probably needed if you want to run on large problems, which is out of scope of mompy. 
But it is really easy to try this technique on small problems
where we want to minimize f subject to constraints g:

```python
x,y = sp.symbols('x,y')
f =  x**2 + y**2
gs = [x+y>=4, x+y<=4]
sol = mp.solvers.solve_GMP(f, gs)
```

We also support direct moment constraints in GMP, see the worksheets.

### Related software and their descriptions

|Software | paper|
|---------|--------|
|[GloptiPoly (MATLAB)](http://homepages.laas.fr/henrion/software/gloptipoly/) | [GloptiPoly 3: moments, optimization and semidefinite programming](http://homepages.laas.fr/henrion/papers/gloptipoly.pdf)|
[SOSTOOLS (MATLAB)](http://www.cds.caltech.edu/sostools/) | [Introducing SOSTOOLS: A General Purpose Sum of Squares Programming Solver](http://www.cds.caltech.edu/~doyle/hot/CDC02_2.pdf)|
|[ncpol2sdpa (python)](https://pypi.python.org/pypi/ncpol2sdpa/) | [Algorithm 950: Ncpol2sdpa](http://arxiv.org/abs/1308.6029)|

### References

J. B. Lasserre. Global optimization with polynomials and the problem of moments. SIAM Journal on Optimization, 11(3):796–817, 2001

J. B. Lasserre. Moments, Positive Polynomials and Their Applications. Imperial College Press, 2011.

J. B. Lasserre. A semidefinite programming approach to the generalized problem of moments. Mathematical Programming, 112(1):65–92, 2008.

P. A. Parrilo and B. Sturmfels. Minimizing polynomial functions. Algorithmic and quantitative real algebraic geometry, DIMACS Series in Discrete Mathematics and Theoretical Computer Science, 60:83–99, 2003

P. A. Parrilo. Semidefinite programming relaxations for semialgebraic problems. Mathematical programming, 96(2):293–320, 2003.

D. Henrion and J. Lasserre. Detecting global optimality and extracting solutions in GloptiPoly. In Positive
polynomials in control, pages 293–310, 2005 [pdf](http://homepages.laas.fr/henrion/papers/extract.pdf)

Our paper is here
```
@inproceedings{wang2015polynomial, 
author = {Sida I. Wang and Arun Chaganty and Percy Liang}, 
booktitle = {Advances in Neural Information Processing Systems (NIPS)}, 
title = {Estimating Mixture Models via Mixture of Polynomials}, 
year = {2015} } 
```
