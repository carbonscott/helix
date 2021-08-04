## Install the package

```
pip install git+https://github.com/carbonscott/helix.git --upgrade --user
```


## Install `lmfit` -- the minimizer

```
pip install git+https://github.com/lmfit/lmfit-py.git --upgrade --user
```


## Characterize a helix using Frenet (TNB) frame

- [Construct TNB frame](https://www.integreat.ca/NOTES/CALC/14.06.html)
- [Derive radius](https://link.springer.com/10.1007/s00214-009-0639-4)
- [B-spline curve using scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html)

### B-spline fitting results

The fitting gives rise to model points that are really close to the original
data.  

![](./examples/helix.tnb.png)


### Helical properties derived from B-spline fitting

![](./examples/1mxr.helix.png)


## Parameterize a helix

A helix is parameterized as `h(t) = [x(t), y(t), z(t)]` with a parameter `t`, in which
`t` is a bounded integer.  So a derivative like `dx/dt` can be achieved using
`np.gradient(x)`.  

### How does `np.gradient` work?

```
np.gradient(x)
```

is similar to

```
dt = 1
(x[2:] - x[:-2]) / (2 * dt)
```

except for the value on the boundary.  
