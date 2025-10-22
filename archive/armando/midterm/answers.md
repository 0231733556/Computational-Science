# Exercise 3 answers

a. What is the difference between continuous optimization and discrete optimization?
> Continuous optimization works with functions that accept real values, thus it allows us to use calculus to find gradients, hessians and thus use algorithms like gradient descent and newton's method. 
Discrete optimization works with functions that accept integer values, thus calculus doesn't apply, and we often have to use combinatorics to solve these optimization problems, which is much harder than using calculus.
---
b. Give two advantages of using sparse matrices in computing algorithms
> Sparse matrices take less memory (you can "bundle" 0s together) and operations over them (like matrix-vector multiplication) are often much faster. Additionally, there are special factorizations of sparse matrices that can be used.
---
c. Give a disadvantage of using metaheuristic optimization algorithms over algorithms for continuous optimization as we have seen in the course.
> Metaheuristic optimization algorithms provide no guarantee of optimality or fast convergence. They also add tuning overhead, whereas the continuous optimization problems are less general but they provide better guarantees and and faster convergence.
--- 
d. What is the purpose of the Armijo rule?
> The Armijo rule is used in backtracking (and conjugate) gradient descent. It ensures sufficient descent by shrinking alpha until the armijo inequality holds, at which point we know we have reached an alpha that provides a sufficient descent. Doing this stabilizes descent and helps convergence.
---
e. Provide the analytical expression of the gradient at the fourth row for the following objective function: m(u) = u_1^2 * u_2^2 * u_3^2 * u_4^2 * u_5^2
> The 4th row component of the gradient of m(u) is given by:

    d m/d u_4 = 2*u_1 * u_2^2 * u_3^2 * u_4^2 * u_5^2
---