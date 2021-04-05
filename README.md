# DynamicBoundspODEsDiscrete.jl
Parametric Discretize-and-Relax methods within DynamicBounds.jl

| **Linux/OS/Windows**                                   |        **Coverage**             |              
|:-------------------------------------------------------:|:-------------------------------------------------------:|
| [![Build Status](https://github.com/PSORLab/DynamicBoundspODEsDiscrete.jl/workflows/CI/badge.svg?branch=master)](https://github.com/PSORLab/DynamicBoundspODEsDiscrete.jl/actions?query=workflow%3ACI) | [![codecov](https://codecov.io/gh/PSORLab/DynamicBoundspODEsDiscrete.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PSORLab/DynamicBoundspODEsDiscrete.jl)) |

## Summary
This package implements a discretize-and-relax approaches to
computing state bounds and relaxations using the DynamicBounds.jl framework. These methods discretize the time domain over into a finite number of points and then compute valid
relaxations at these time-points. Full documentation of this functionality may be found [here](https://psorlab.github.io/DynamicBounds.jl/dev/pODEsDiscrete/pODEsDiscrete) in the DynamicBounds.jl website.

## Installation

```julia
using Pkg; Pkg.add("DynamicBoundspODEsDiscrete")
```

or using the following command in the pacakge manager environment
```
pkg > add DynamicBoundspODEsDiscrete
```

Note that this package can also be used directly via DynamicBounds.jl as the later
package automatically reexports it.

## References
- Corliss, G. F., & Rihm, R. (1996). Validating an a priori enclosure using high-order Taylor series. MATHEMATICAL RESEARCH, 90, 228-238.
- Lohner, R. J. (1992, January). Computation of guaranteed enclosures for the solutions of ordinary initial and boundary value problems. In Institute of mathematics and its applications conference series (Vol. 39, pp. 425-425). Oxford University Press.
- Nedialkov, Nedialko S., and Kenneth R. Jackson. "An interval Hermite-Obreschkoff method for computing rigorous bounds on the solution of an initial value problem for an ordinary differential equation." Reliable Computing 5.3 (1999): 289-310.
- Nedialkov, Nedialko Stoyanov. Computing rigorous bounds on the solution of an initial value problem for an ordinary differential equation. University of Toronto, 2000.
- Nedialkov, N. S., & Jackson, K. R. (2000). ODE software that computes guaranteed bounds on the solution. In Advances in Software Tools for Scientific Computing (pp. 197-224). Springer, Berlin, Heidelberg.
- Sahlodin, A. M., & Chachuat, B. (2011). Discretize-then-relax approach for convex/concave relaxations of the solutions of parametric ODEs. Applied Numerical Mathematics, 61(7), 803-820.
- Wilhelm, M. E., Le, A. V., & Stuber, M. D. (2019). Global optimization of stiff dynamical systems. AIChE Journal, 65(12), e16836
