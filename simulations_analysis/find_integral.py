import numpy as np
import matplotlib.pyplot as plt

N = 1000
K = 50

def phi(s, a, period):
    return -a * period / (2 * np.pi) * np.cos(2 * np.pi * s / period)

def integ_cos_phi(a, period):
    s = np.linspace(0, period, N, endpoint=True)
    cos_phi = np.cos(phi(s, a, period))
    return np.trapz(cos_phi, s) / period

# plot vs a for period = 2*pi
a = np.linspace(0, 1, K)
integral = np.zeros(K)
for i in range(K):
    integral[i] = integ_cos_phi(a[i], 2 * np.pi)
plt.plot(a, integral)
# fit u for e^(a*u)
u = np.log(integral[1:]) / a[1:]
u = np.mean(u)
print(u)
# plot fit
plt.plot(a, np.exp(a * u))
plt.savefig('integral_a.png')
plt.close()

# plot vs period for a = 1
period = np.linspace(0.1, 10, K)
integral = np.zeros(K)
for i in range(K):
    integral[i] = integ_cos_phi(1, period[i])
plt.plot(period, integral)
plt.savefig('integral_period.png')
plt.close()