# Ecuaciones lineales diofantinas aplicadas a programas lineales enteros

*Linear Diophantine Equations applied to Integer Linear Programs*

This repository contains my undegraduate thesis, submitted to the Instituto Tecnológico Autónomo de
México (ITAM) as a requirement for the Bachelor of Science degree in Applied Mathematics.

The thesis focuses on ILP instances where a constraint is orthogonal to the objective vector, a
scenario in which traditional Branch-and-Cut methods become extremely inefficient. By establishing a
theoretical equivalence between such ILPs and solving related linear diophantine equations, this
work develops novel algorithms that avoid exhaustive search and significantly improve solution
times.

----
When an ILP's constraint is orthogonal to the objetive direction, Branch-and-Cut methods are unable
to efficiently prune the search tree. In these cases, the method may generate exponentially deep
trees (or even infinitely deep trees), practically and theoretically rendering them unable to solve
those types of instances.

The thesis proves that finding an optimal integer solution for the problematic ILP
is equivalent to finding integer solutions to certain diophantine equations derived from the
constraint. The coefficients of these equations come from a coprime vector $q$ associated with
the objective vector $p$. The sign pattern of $q$ divides this ILP class into two cases:

    - $q$ contains a negative entry. The optimum can be found by solving a single Diophantine
      equation. An algorithm is presented that constructs the optimal solution in polynomial time
      (quadratic in problem dimension) Moreover, the optimal objective value can be identified in
      $O(n^2\log_2||p||_\intfy)$ time without explicitly solving the ILP for rational objective vectors
      $p$.

    - All entries of $q$ are non-negative. A finite number of diophantine equations must be solved.
      However, if the right hand side of the orthogonal constraint is sufficiently large, it
      suffices to solve a single linear diophantine equation. As a byproduct of the techniques
      developed for this case, new upper bounds for the classical Frobenius coin-change problem are
      obtained. An algorithm is provided that outperforms Branch-and-Cut methods on these instances.

Extensive computational experiments show that both specialized algorithms consistently outperform
the open-source COIN-OR Branch-and-Cut solver on targeted instances. These results confirm the
practical advantages of our diophantine-based approach.

Finally, the thesis sketches an extension to general ILPs by searching for integer solutions to
systems of linear diophantine equations. This reduces ILPs to systems of linear integer
inequalities, suggesting a possible research direction for future work.
----

This work will be of interest to researchers in integer programming and optimization, particularly
those exploring alternatives to Branch-and-Cut. The novel perspective of
tackling ILPs via diophantine equations could inspire further research or applications in
specialized ILP solvers.

For questions or collaborative interests, please open an issue or contact me via email.
