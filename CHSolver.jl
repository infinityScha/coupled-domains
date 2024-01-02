using Statistics
using Plots
using ProgressMeter
using FFTW

# Semi-implicit Fourier spectral method for the Cahn-Hilliard equation 
# following https://www.sciencedirect.com/science/article/pii/S001046559700115X

# physical parameters
const γ = 1.0 # line tension [kT/nm] = K*4/(3*sqrt(2K/ϵ)) = 8*K/3*σ -> K = 3*γ/8/σ 
const σ = 0.8 # interface width [nm]
const ζ = 0.01 # Cook noise [kT/ns]
const m = 1.0 # mobility [nm^2/ns/kT]
const c₀ = 0.0 # bulk field value
# numerical parameters
const M = 6 # log(2, grid size)
const L = 10.0 # box size [nm]
const Δt = 1.0e-2 # time step [ns]
# deduced numerical parameters
const N = 2^M # grid size
# deduced physical parameters
# TODO: fix in general + fix K, ϵ definition
const K = 3*γ/8/σ
const ϵ = 8*K/σ*σ
# deduced Fourier parameters
k = 2.0*π/L*cat(0:N/2-1, -N/2:-1, dims=1)
k² = zeros(N,N)
for i in 1:N
    for j in 1:N
        k²[i,j] = k[i]^2 + k[j]^2
    end
end
k⁴ = k².^2
# deduced semi-implicit parameters
temp = 1.0 .+ m*K*Δt*k⁴
const A = 1.0./temp
println(maximum(A.*temp), " ", minimum(A.*temp))
const B = -m*ϵ*Δt*k²./temp
const η = m*sqrt(ζ)
println("K = ", K, " [kT/nm^2]", ", ϵ = ", ϵ, " [kT/nm^2]")

function solver(c, Nsteps, save_every=100)
    c_list = [c]
    @showprogress for i in 1:Nsteps
        # Fourier transform
        ĉₙ = fft(c)
        # random noise
        f̂ = η*randn(N, N)
        f̂ .-= mean(f̂)
        f̂ = fft(f̂)
        # compute nonlinear term
        ĉₙ₊₁ = A.*(ĉₙ .+ f̂) .+ B.*fft(-c.+c.^3)
        # inverse Fourier transform
        c = real(ifft(ĉₙ₊₁))
        # save
        if i%save_every == 0
            push!(c_list, c)
        end
    end
    return c_list
end

function animate_time_evolution(c_list)
    anim = @animate for i in 1:length(c_list)
        heatmap(c_list[i], aspect_ratio=1, color=:jet, cbar=false, clim=(-1,1))
        # remove axes and ticks
        plot!(size=(500,500), ticks=nothing, border=:none, axis=nothing)
    end
    gif(anim, "CHSolver.gif", fps=10)
end


# make initial uniform grid + random uniform noise
c = 2.0*rand(N,N) .- 1.0
c .+= c₀ - mean(c)
# solve
c_list = solver(c, 10000, 100)
#print([c_list[i][1,1] for i in 1:length(c_list)])
animate_time_evolution(c_list)