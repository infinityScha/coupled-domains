using Statistics
using Plots
using ProgressMeter
using FFTW

# set GR plotting backend
gr()

# Semi-implicit Fourier spectral method for the Cahn-Hilliard equation 
# following https://www.sciencedirect.com/science/article/pii/S001046559700115X

# physical parameters
const γ = 0.5 # line tension [kT/nm] = K*4/(3*sqrt(2K/ϵ)) = 8*K/3*σ -> K = 3*γ/8/σ 
const σ = 1.2 # interface width [nm]
const λ = 0.03 # interleaflet coupling [kT/nm^2]
const δ = 0.01 # sinusoidal perturbation amplitude kT/nm^2
const n = 1 # field periodicity
const ζ = 0.025 # Cook noise [kT/ns]
const m = 1.0 # mobility [nm^2/ns/kT]
const c₀ = 0.0 # bulk field value
# numerical parameters
const M = 6 # log(2, grid size)
const L = 20.0 # box size [nm]
const Δt = 2.5e-2 # time step [ns]
# deduced numerical parameters
const N = 2^M # grid size
# deduced physical parameters
# TODO: fix in general + fix K, ϵ definition + validate with 1D
const K = 3*γ/8/σ
const ϵ = 8*K/σ*σ
const Λ = 0.25*λ
# external field matrix
const H = zeros(N,N)
for i in 1:N
    for j in 1:N
        H[i,j] = δ*sin(2*π*n*i/N)
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

function plot_spectrum(cᵤ, cₗ, coupling_list)
    # plot on p
    heatmap1 = heatmap(cᵤ, aspect_ratio=1, color=:jet, cbar=false, clim=(-1,1), axis=nothing, border=:none, ticks=nothing, title = "Upper leaflet")
    heatmap2 = heatmap(cₗ, aspect_ratio=1, color=:jet, cbar=false, clim=(-1,1), axis=nothing, border=:none, ticks=nothing, title = "Lower leaflet")
    coupling_plot = plot(coupling_list, xlims=(1, length(coupling_list)), ylims=(0,1.0), color=:black, legend=false, title="Coupling ratio")
    p = plot(heatmap1, heatmap2, coupling_plot, layout=(1,3), size=(1500,500))
    return p
end

function coupling_ratio(cᵤ, cₗ)
    return 0.5.*mean(abs.(cᵤ .+ cₗ))
end

function noise()
    f̂ = η*randn(N, N)
    f̂ .-= mean(f̂)
    f̂ = fft(f̂)
    return f̂
end

function solver(cᵤ, cₗ, Nsteps, ;save_every=100, plot_every=0)
    c_list = [[cᵤ, cₗ]]
    coupling_list_save = [coupling_ratio(cᵤ, cₗ)]
    coupling_list_plot = [coupling_ratio(cᵤ, cₗ)]
    @showprogress for i in 1:Nsteps
        # Fourier transform
        ĉⁿᵤ = fft(cᵤ)
        ĉⁿₗ = fft(cₗ)
        # random noise
        f̂ᵤ, f̂ₗ = noise(), noise()
        # compute nonlinear term
        ĉⁿ⁺¹ᵤ = A.*(ĉⁿᵤ .+ f̂ᵤ) .+ B.*fft(-cᵤ.+cᵤ.^3 .- Λ*(cₗ .- cᵤ) .+ H)
        ĉⁿ⁺¹ₗ = A.*(ĉⁿₗ .+ f̂ₗ) .+ B.*fft(-cₗ.+cₗ.^3 .- Λ*(cᵤ .- cₗ) .- H)
        # inverse Fourier transform
        cᵤ = real(ifft(ĉⁿ⁺¹ᵤ))
        cₗ = real(ifft(ĉⁿ⁺¹ₗ))
        # save
        if (save_every != 0) && (i % save_every == 0)
            push!(coupling_list_save, coupling_ratio(cᵤ, cₗ))
            push!(c_list, [cᵤ, cₗ])
        end
        if (plot_every != 0) && (i % plot_every == 0)
            push!(coupling_list_plot, coupling_ratio(cᵤ, cₗ))
            p = plot_spectrum(cᵤ, cₗ, coupling_list_plot)
            display(p)
        end
    end
    return c_list, coupling_list_save
end

function animate_time_evolution(c_list, coupling_list)
    anim = @animate for i in 1:length(c_list)
        cᵤ, cₗ = c_list[i]
        p = plot_spectrum(cᵤ, cₗ, coupling_list[1:i])
    end
    gif(anim, "CHSolver.gif", fps=10)
end

function plot_coupling_list(coupling_list)
    plot(coupling_list, xlims=(1, length(coupling_list)), ylims=(0,1.0), color=:black, legend=false, title="Coupling ratio")
    savefig("coupling_list.png")
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
c_list, coupling_list = solver(cᵤ, cₗ, 500000, save_every=1000, plot_every=0)
animate_time_evolution(c_list, coupling_list)