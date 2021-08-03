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

![](./examples/helix.tnb.png)


### Helical properties derived from B-spline fitting

![](./examples/1mxr.helix.png)
