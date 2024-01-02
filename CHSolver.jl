using Statistics
using Plots
using ProgressMeter
using FFTW

# Semi-implicit Fourier spectral method for the Cahn-Hilliard equation 
# following https://www.sciencedirect.com/science/article/pii/S001046559700115X

# physical parameters
const γ = 1.0 # line tension [kT/nm] = K*4/(3*sqrt(2K/ϵ)) = 8*K/3*σ -> K = 3*γ/8/σ 
const σ = 0.8 # interface width [nm]
const λ = 0.01 # interleaflet coupling [kT/nm^2]
const δ = 0.0 #0.1 # sinusoidal perturbation amplitude kT/nm^2
const ζ = 0.05 # Cook noise [kT/ns]
const m = 1.0 # mobility [nm^2/ns/kT]
const c₀ = 0.0 # bulk field value
# numerical parameters
const M = 5 # log(2, grid size)
const L = 10.0 # box size [nm]
const Δt = 1.0e-2 # time step [ns]
# deduced numerical parameters
const N = 2^M # grid size
# deduced physical parameters
# TODO: fix in general + fix K, ϵ definition
const K = 3*γ/8/σ
const ϵ = 8*K/σ*σ
const Λ = 0.5*λ
# external field matrix
const H = zeros(N,N)
for i in 1:N
    for j in 1:N
        H[i,j] = δ*sin(4*π*i/N)
    end
end
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
const B = -m*ϵ*Δt*k²./temp
const η = m*sqrt(ζ)
println("K = ", K, " [kT/nm^2]", ", ϵ = ", ϵ, " [kT/nm^2]")

function noise()
    f̂ = η*randn(N, N)
    f̂ .-= mean(f̂)
    f̂ = fft(f̂)
    return f̂
end

function solver(cᵤ, cₗ, Nsteps, save_every=100)
    c_list = [[cᵤ, cₗ]]
    @showprogress for i in 1:Nsteps
        # Fourier transform
        ĉⁿᵤ = fft(cᵤ)
        ĉⁿₗ = fft(cₗ)
        # random noise
        f̂ᵤ, f̂ₗ = noise(), noise()
        # compute nonlinear term
        ĉⁿ⁺¹ᵤ = A.*(ĉⁿᵤ .+ f̂ᵤ) .+ B.*fft(-cᵤ.+cᵤ.^3 .- Λ*(cₗ .- cᵤ) .+ H.*cᵤ)
        ĉⁿ⁺¹ₗ = A.*(ĉⁿₗ .+ f̂ₗ) .+ B.*fft(-cₗ.+cₗ.^3 .- Λ*(cᵤ .- cₗ) .- H.*cₗ)
        # inverse Fourier transform
        cᵤ = real(ifft(ĉⁿ⁺¹ᵤ))
        cₗ = real(ifft(ĉⁿ⁺¹ₗ))
        # save
        if i%save_every == 0
            push!(c_list, [cᵤ, cₗ])
        end
    end
    return c_list
end

function animate_time_evolution(c_list)
    anim = @animate for i in 1:length(c_list)
        # plot heatmaps for both leaflets and external field
        plot(heatmap(c_list[i][1], aspect_ratio=1, color=:jet, cbar=false, clim=(-1,1)), 
            heatmap(c_list[i][2], aspect_ratio=1, color=:jet, cbar=false, clim=(-1,1)), 
            heatmap(H, aspect_ratio=1, color=:jet, cbar=false, clim=(-δ-1.0e-8,δ+1.0e-8)),
            layout=(1,3), size=(1500,500), ticks=nothing, border=:none, axis=nothing)
    end
    gif(anim, "CHSolver.gif", fps=10)
end


# make initial uniform grid + random uniform noise
#cᵤ = 2.0*rand(N,N) .- 1.0
#cₗ = 2.0*rand(N,N) .- 1.0
cᵤ = zeros(N,N)
cₗ = zeros(N,N)
# set upper half to 1, lower half to -1
cᵤ[1:N÷2,:] .= -1.0
cₗ[1:N÷2,:] .= 1.0
cᵤ[N÷2+1:end,:] .= 1.0
cₗ[N÷2+1:end,:] .= -1.0

cᵤ .+= c₀ - mean(cᵤ)
cₗ .+= c₀ - mean(cₗ)
# solve
c_list = solver(cᵤ, cₗ, 500000, 500)
animate_time_evolution(c_list)