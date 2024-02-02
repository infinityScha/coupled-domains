using Statistics
using Plots
using ProgressMeter
using FFTW
using Printf
using JLD2

# set GR plotting backend
gr()

# Semi-implicit Fourier spectral method for the Cahn-Hilliard equation 
# following https://www.sciencedirect.com/science/article/pii/S001046559700115X

# physical parameters
const γ = 0.5 # line tension [kT/nm] = K*4/(3*sqrt(2K/ϵ)) = 8*K/3*σ -> K = 3*γ/8/σ 
const σ = 2.0 # interface width [nm]
const λ = 0.15 # interleaflet coupling [kT/nm^2]
const dλdt = 0.0 #-0.002 # interleaflet coupling time derivative [kT/nm^2/ns]
const Kᴸᵒ = 22.9 # bending modulus of Lo phase [kT]
const Kᴸᵈ = 8.9 # bending modulus of Ld phase [kT]
const R = 10.0#10.0 # radius of curvature at the maximum [nm] 
const Jₛᴸᵒ = -0.178 # spontaneous curvature of Lo phase [nm^-1]
const Jₛᴸᵈ = 0.114 # spontaneous curvature of Ld phase [nm^-1]
const ζ = 2.5e-7 # Cook noise [kT/ns]
const m = 1.0 # mobility [nm^2/ns/kT]
const c₀ = 0.0 # bulk field value
# numerical parameters
const M = 6 # log(2, grid size)
const L = 20.0 # box size [nm]
const Δt = 2.5e-3 # time step [ns]
# deduced numerical parameters
const N = 2^M # grid size
# deduced physical parameters
# TODO: fix in general + fix K, ϵ definition + validate with 1D
const ϵ = γ/2/σ
const K = σ*σ*ϵ/2
# external field matrix
const H₁ = zeros(N,N)
const H₂ = zeros(N,N)
# approximate curvature matrix
function Hij(i, R)
    c = sin(2*π*i/N)/R
    enᴸᵒ = 0.5*Kᴸᵒ*(c + Jₛᴸᵒ)^2 # at Φ = 1
    enᴸᵈ = 0.5*Kᴸᵈ*(c + Jₛᴸᵈ)^2 # at Φ = -1
    return 0.5*(enᴸᵒ - enᴸᵈ)
end

for i in 1:N
    for j in 1:N
        H₁[i,j] = Hij(i, R)
        H₂[i,j] = Hij(i, -R)
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
const η = m*sqrt(ζ/Δt)/L*N
println("K = ", K, " [kT]", ", ϵ = ", ϵ, " [kT/nm^2]")

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

function save_state(cᵤ, cₗ, filename="CHSolver.jld2")
    # save state
    save(filename, "cᵤ", cᵤ, "cₗ", cₗ)
end

function load_state(filename="CHSolver.jld2")
    # load state
    cᵤ = load(filename, "cᵤ")
    cₗ = load(filename, "cₗ")
    return cᵤ, cₗ
end

function noise()
    f̂ = Δt*η*randn(N, N)
    f̂ .-= mean(f̂)
    f̂ = fft(f̂)
    return f̂
end

function solver(cᵤ, cₗ, Nsteps, ;save_every=100, plot_every=0)
    λ_ = λ
    c_list = [[cᵤ, cₗ]]
    coupling_list_save = [coupling_ratio(cᵤ, cₗ)]
    coupling_list_plot = [coupling_list_save[end]]
    pbar = Progress(Nsteps, desc=@sprintf("%.3f", coupling_list_save[end]))
    for i in 1:Nsteps
        # Fourier transform
        ĉⁿᵤ = fft(cᵤ)
        ĉⁿₗ = fft(cₗ)
        # random noise
        f̂ᵤ, f̂ₗ = noise(), noise()
        # more BS
        cᵤ⁵ = cᵤ.^5
        cₗ⁵ = cₗ.^5
        # compute nonlinear term
        ĉⁿ⁺¹ᵤ = A.*(ĉⁿᵤ .+ f̂ᵤ) .+ B.*fft(0.25*12.0.*cᵤ⁵.*(cᵤ⁵.*cᵤ .- 1.0) .- 0.25*λ_*(cₗ .- cᵤ).+ H₁)   
        ĉⁿ⁺¹ₗ = A.*(ĉⁿₗ .+ f̂ₗ) .+ B.*fft(0.25*12.0.*cₗ⁵.*(cₗ⁵.*cₗ .- 1.0) .- 0.25*λ_*(cᵤ .- cₗ).+ H₂)
        #ĉⁿ⁺¹ᵤ = A.*(ĉⁿᵤ .+ f̂ᵤ) .+ B.*fft(-cᵤ .+ cᵤ.^3 .- 0.25*λ_*(cₗ .- cᵤ).+ H₁) 
        #ĉⁿ⁺¹ₗ = A.*(ĉⁿₗ .+ f̂ₗ) .+ B.*fft(-cₗ .+ cₗ.^3 .- 0.25*λ_*(cᵤ .- cₗ).+ H₂)
        # inverse Fourier transform
        cᵤ = real(ifft(ĉⁿ⁺¹ᵤ))
        cₗ = real(ifft(ĉⁿ⁺¹ₗ))
        # save
        if (save_every != 0) && (i % save_every == 0)
            push!(coupling_list_save, coupling_ratio(cᵤ, cₗ))
            push!(c_list, [cᵤ, cₗ])
            # set progressbar title to coupling ratio
            pbar.desc = @sprintf("%.3f %.3f %.3f", coupling_list_save[end], λ_, mean(cᵤ))
        end
        if (plot_every != 0) && (i % plot_every == 0)
            push!(coupling_list_plot, coupling_ratio(cᵤ, cₗ))
            p = plot_spectrum(cᵤ, cₗ, coupling_list_plot)
            display(p)
            pbar.desc = @sprintf("%.3f %.3f %.3f", coupling_list_plot[end], λ_, mean(cᵤ))
        end
        next!(pbar)
        λ_ = λ_ > 0.0 ? λ_ + dλdt*Δt : 0.0
    end
    return c_list, coupling_list_save
end

function animate_time_evolution(c_list, coupling_list, filename="CHSolver.gif")
    anim = @animate for i in eachindex(c_list)
        p = plot_spectrum(c_list[i][1], c_list[i][2], coupling_list[1:i])
    end
    gif(anim, filename, fps=10)
end

function plot_coupling_list(coupling_list)
    plot(coupling_list, xlims=(1, length(coupling_list)), ylims=(0,1.0), color=:black, legend=false, title="Coupling ratio")
    savefig("coupling_list.png")
end


function make_grid(c₀, upper=true)
    # make initial zero grid
    c = zeros(N,N)
    if upper
        # set 0.5+(1.0+c₀) of upper as 1.0
        i = round(Int, 0.5*(1.0+c₀)*N)
        c[1:i,:] .= 1.0
        # set the rest to -1.0
        c[i+1:end,:] .= -1.0
    else
        # set 0.5+(1.0-c₀) of upper as -1.0
        i = round(Int, 0.5*(1.0-c₀)*N)
        c[1:i,:] .= -1.0
        # set the rest to 1.0
        c[i+1:end,:] .= 1.0
    end
    c .+= c₀ - mean(c)
    return c
end

# make initial uniform grid
#cᵤ = make_grid(c₀, true)
#cₗ = make_grid(c₀, true)
cᵤ, cₗ = load_state("final_large.jld2")

# solve
#c_list, coupling_list = solver(cᵤ, cₗ, 1000000, save_every=0, plot_every=25)
c_list, coupling_list = solver(cᵤ, cₗ, 1000000, save_every=1000, plot_every=50)
#c_list, coupling_list = solver(cᵤ, cₗ, 100000, save_every=10000, plot_every=0)
save_state(c_list[end][1], c_list[end][2], "final_large2.jld2")
animate_time_evolution(c_list, coupling_list, "large2.gif")