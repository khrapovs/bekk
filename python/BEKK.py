import numpy as np

# BEKK model
# u(t)|H(t) ~ N(0,H(t))
# H(t) = E_{t-1}[u(t)u(t)']
# One lag, no asymmetries
# H(t) = CC' + Au(t-1)u(t-1)'A' + BH(t-1)B'