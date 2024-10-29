import numpy as np
import matplotlib.pyplot as plt
# set all plt font sizes to 15
plt.rcParams.update({'font.size': 17})

R = 5.0 # Radius of curvature
Lx = 34.0

def k(s0, s1, N=1000): # Curvature as a function of s
    s = np.linspace(s0, s1, N)
    return np.sin(2*np.pi*s/Lx) / R

def phi(ss, ks): # cummualtive integral of sin(2pi*x/Lx)/R 
    ds = ss[1] - ss[0]  # Assume uniform spacing
    phis = np.cumsum(ks) * ds 
    phis -= np.mean(phis) # Normalize
    return phis

def r(ss, phis): # cummualtive integral of (cos phi, sin phi)
    dx = np.cos(phis)
    dy = np.sin(phis)
    ds = ss[1] - ss[0]  # Assume uniform spacing

    x = np.cumsum(dx)*ds
    y = np.cumsum(dy)*ds
    return x, y

# Generate the data
print("Generating data...")
x0 = 0.0
x1 = Lx
N = 10000
x = np.linspace(x0, x1, N)
ks = k(x0, x1, N)
phis = phi(x, ks)
xs, ys = r(x, phis)


# Plot the data, make two differrnt y axes
print("Plotting data...")
fig, ax = plt.subplots()
ax.plot(xs, ys, 'b-')
ax.set_xlabel('x')
ax.set_ylabel('y', color='b')
ax.tick_params('y', colors='b')

ax2 = ax.twinx()
ax2.plot(xs, ks, 'r-')
ax2.set_ylabel('Radius of curvature', color='r')
ax2.tick_params('y', colors='r')
ax2.locator_params(axis='y', nbins=5)


ax.set_xlim(0.0, xs[-1])
ax.set_ylim(np.min(ys) - 1, np.max(ys) + 1)
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig("curve_figure.png", dpi=300)
plt.close()