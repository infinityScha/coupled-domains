using Statistics
using Plots
using ProgressMeter
using Printf
using FFTW

# Solver for the 1D Cahn-Hilliard equation with a given chemical potential function.
# The aim is to fine the integration constant for said function. 

# set GR plotting backend
gr()

# physical parameters
const ϵ = 2.0
const K = ϵ # DO NOT CHANGE
const M = 12 # log(2, grid size)
const L = 30.0 # box size [nm]
const Δt = 1.0e-3 # time step [ns]
# deduced numerical parameters
const N = 2^M # grid size

# deduced Fourier parameters
k = 2.0*π/L*cat(0:N/2-1, -N/2:-1, dims=1)
k² = k.^2
k⁴ = k².^2

# deduced semi-implicit parameters
temp = 1.0 .+ K*Δt*k⁴
const A = 1.0./temp
const B = -ϵ*Δt*k²./temp

# chemical potential function (DEFINE HERE!)
function μ(c)
    #c³ = c.^3
    #return 0.25*8.0.*c³.*(c³.*c .- 1.0)
    c⁵ = c.^5
    return 0.25*12.0.*c⁵.*(c⁵.*c .- 1.0)
    #c⁷ = c.^7
    #return 0.25*16.0.*c⁷.*(c⁷.*c .- 1.0)
    #c¹¹ = c.^11
    #return 0.25*24.0.*c¹¹.*(c¹¹.*c .- 1.0)
    #return -c .+ c.^3 
end

function solver(c, Nsteps)
    pbar = Progress(Nsteps)
    for i in 1:Nsteps
        # Fourier transform
        ĉⁿ = fft(c)
        # compute nonlinear term
        ĉⁿ⁺¹ = A.*ĉⁿ .+ B.*fft(μ(c))
        # inverse Fourier transform
        c = real(ifft(ĉⁿ⁺¹))
        next!(pbar)
    end
    return c
end

function plot_spectrum(c)
    # plot on p
    p = plot(range(0.0, L, N), c, size=(500,500))
    # plot tanh on p
    plot!(p, range(0.0, L, N), tanh.(range(-L/2, L/2, N)/sqrt(2)), lw=2)
    plot!(p, range(0.0, L, N), tanh.(range(-L/2, L/2, N)), lw=2)
    display(p)
end

function get_I(c)
    I = 0.0
    for i in 1:N
        I += ((c[i] - c[i%N+1]))^2
    end 
    println("I = ", 0.5*I/L*N./sqrt(2))
end

# initial condition
c = zeros(N)
c[1:N÷2] .= -1.0
c[N÷2+1:N] .= 1.0
# solve

function multi_solve(c,n,m)
    for i in 1:n
        c = solver(c, m)
        # get I
        get_I(c)
    end
    return c
end

# plot
plot_spectrum(multi_solve(c,10,10000))
