using Statistics
using Plots
using ProgressMeter
using FFTW
using Printf
using JLD2
using ColorSchemes
using Colors
using KernelDensity

# set GR plotting backend
ENV["GKSwstype"] = "100"
gr()

# Semi-implicit Fourier spectral method for the Cahn-Hilliard equation 
# following https://www.sciencedirect.com/science/article/pii/S001046559700115X

# physical parameters
const I = 2/3*sqrt(2) #0.85714*sqrt(2) # for f(c)=0.25*(c⁶-1)², for regular CH it is 2/3*sqrt(2)
const γ = 2.0 # line tension [kT/nm] = I√(K⋅ϵ)
const σ = 0.8 # interface width [nm] = √(K/ϵ)
const ζ = 1.0 # Cook noise [kT/ns]
const max_noise_wavelength = 2.0 # [nm]
const Δx_mc = 2.5e-2 # monte carlo step size
const c₀ = 0.0 # bulk field value
# numerical parameters
const M = 5 # log(2, grid size)
const L = 24.0 #max_noise_wavelength * (2^M) # box size [nm]
const Δt = 5.0e-3 # time step [ns]
# deduced numerical & physical parameters
const N = 2^M # grid size
const K = γ*σ/I
const ϵ = K/σ^2

# deduced Fourier parameters
k = (2.0*π/L)*cat(0:N/2-1, -N/2:-1, dims=1)
k_vecs = zeros(N,N,2)
for i in 1:N
    for j in 1:N
        k_vecs[i,j,1] = k[i]
        k_vecs[i,j,2] = k[j]
    end
end
k² = zeros(N,N)
for i in 1:N
    for j in 1:N
        k²[i,j] = k[i]^2 + k[j]^2
    end
end
k⁴ = k².^2

# deduced semi-implicit parameters
temp = 1.0 .+ K*Δt*k⁴
const A = 1.0./temp
const B = -ϵ*Δt*k²./temp
const C = -Δt./temp
const η = sqrt(2.0*ζ/Δt)/L*N

println("K = ", K, " [kT]", ", ϵ = ", ϵ, " [kT/nm^2]")

# Define a custom color map with a gradient of 5 colors
const mycolors = range(RGB(0.4,0.6,1), stop=RGB(1,0.6,0.4), length=10)

# Create a function to map values of c to colors
function colorfunc(c)
    # find in which interval c falls between -1 and 1
    idx = max(round(Int, (c+1.0)/2.0*9) + 1, 1)
    idx = min(idx, 10)
    return mycolors[idx]
end


function plot_spectrum(c)
    # Apply the colorfunc to each element of c
    c_colors = colorfunc.(c)

    # Create the heatmap
    hm = heatmap(c_colors, aspect_ratio=1, color=:jet, cbar=false, clim=(-1,1), axis=nothing, border=:none, ticks=nothing)
    p = plot(hm, size=(500,500))
    return p
end

function save_state(c, filename="CHC_CWT.jld2")
    # save state
    save(filename, "c", c)
end

function load_state(filename="CHC_CWT.jld2")
    # load state
    return load(filename, "c")
end

function save_state_list(c_list, filename="CHC_CWT.jld2")
    # save state
    save(filename, "c_list", c_list)
end

function load_state_list(filename="CHC_CWT.jld2")
    # load state
    return load(filename, "c_list")
end

function divergence(v₁, v₂)
    # compute divergence of vector field
    ∂₁v₁, ∂₂v₂ = zeros(N,N), zeros(N,N)
    ∂₁v₁ = real(ifft(im*k_vecs[:,:,1].*fft(v₁)))
    ∂₂v₂ = real(ifft(im*k_vecs[:,:,2].*fft(v₂)))
    return (∂₁v₁ .+ ∂₂v₂)
end

function noise()
    f₁ = randn(N, N)
    f₂ = randn(N, N)
    # compute divergence of random vector field
    f̂ = fft(divergence(f₁, f₂)*η)
    f̂[1,1] = 0.0
    # keep only fluctuations with k > max_noise_wavelength^-1 
    f̂[k² .> (1.001*4.0*π*π/max_noise_wavelength/max_noise_wavelength)] .= 0.0
    return f̂
end

function solver_noise(c, Nsteps, ;save_every=100, plot_every=0, noisy=true)
    c_list = [c]
    pbar = Progress(Nsteps)
    for i in 1:Nsteps
        # Fourier transform
        ĉⁿ = fft(c)
        # random noise
        f̂ = noisy ? noise() : zeros(N, N)
        # compute nonlinear term
        c⁵ = c.^5
        #ĉⁿ⁺¹ = A.*ĉⁿ .+ B.*fft(3.0.*c⁵.*(c⁵.*c .- 1.0)) + C.*f̂
        ĉⁿ⁺¹ = A.*(ĉⁿ .+ Δt.*f̂) .+ B.*fft(-c .+ c.^3) 
        # inverse Fourier transform
        c = real(ifft(ĉⁿ⁺¹))
        # save
        if (save_every != 0) && (i % save_every == 0)
            push!(c_list, c)
        end
        if (plot_every != 0) && (i % plot_every == 0)
            p = plot_spectrum(c)
            savefig("curr.png")
            p = plot_energy_evolution(c_list, "Noise")
            savefig("energy_evolution.png")
        end
        next!(pbar)
    end
    return c_list
end

function diff(v)
    # compute gradient of scalar field 
    # convert to fourier space
    v̂ = fft(v)
    # compute gradient
    ∂₁v = real(ifft(im*k_vecs[:,:,1].*v̂))
    ∂₂v = real(ifft(im*k_vecs[:,:,2].*v̂))
    return ∂₁v, ∂₂v
end

function energy(c)
    # compute energy over LxL box
    #c⁶ = c.^6
    c⁶ = c.*c
    diff_c₁, diff_c₂ = diff(c)
    return (ϵ*0.25*sum((c⁶ .- 1.0).^2) + 0.5*K*sum(diff_c₁.^2 + diff_c₂.^2))*(L/N)^2
end

function solver_mc(c, Nsteps, ;save_every=100, plot_every=0)
    c_list = [c]
    curr_c = copy(c)
    curr_en = energy(c)
    prev_c = copy(c)
    prev_en = curr_en
    accepted = 0
    total_steps = 0
    pbar = Progress(Nsteps)
    for i in 1:Nsteps
        prev_c = copy(curr_c)
        prev_en = curr_en
        ĉⁿ = fft(curr_c)
        # random noise
        curr_c = real(ifft(ĉⁿ + noise()./η.*Δx_mc))
        # compute energy difference
        curr_en = energy(curr_c)
        ΔE = curr_en - prev_en
        if ΔE > 0.0 && exp(-ΔE) < rand()
            curr_c = prev_c
            curr_en = prev_en
        else
            accepted += 1
        end
        total_steps += 1
        # save
        if (save_every != 0) && (i % save_every == 0)
            push!(c_list, curr_c)
            # update progressbar title
            pbar.desc = @sprintf("%.3f", accepted/total_steps)
        end
        if (plot_every != 0) && (i % plot_every == 0)
            p = plot_spectrum(curr_c)
            savefig("curr.png")
            p = plot_energy_evolution(c_list, "MC")
            savefig("energy_evolution.png")
        end
        next!(pbar)
    end
    return c_list
end

function animate_time_evolution(c_list, filename="CHC_CWT.gif")
    anim = @animate for i in eachindex(c_list)
        p = plot_spectrum(c_list[i])
    end
    gif(anim, filename, fps=10)
end

function plot_energy_kernel_density(c_list, label, ; p = nothing)
    # Create the profile
    if p == nothing
        p = plot()
    end
    en_list = [energy(c) for c in c_list]
    # remove outliers (2 std away from the mean)
    en_list = en_list[abs.(en_list .- mean(en_list)) .< 2*std(en_list)]
    kde_ = kde(en_list)
    plot!(p, kde_.x, kde_.density, label=label, linewidth=2)
    return p
end

function plot_c_kernel_density(c_list, label, ; p = nothing)
    # Create the profile
    if p == nothing
        p = plot()
    end
    c_list_ = [vec(c) for c in c_list]
    c_list_ = hcat(c_list_...)
    # remove outliers (2 std away from the mean)
    c_list_ = c_list_[abs.(c_list_ .- mean(c_list_)) .< 2*std(c_list_)]
    kde_ = kde(c_list_)
    plot!(p, kde_.x, kde_.density, label=label, linewidth=2)
    return p
end

function plot_energy_evolution(c_list, label, ; p = nothing)
    # Create the profile
    if p == nothing
        p = plot()
    end
    en_list = [energy(c) for c in c_list]
    plot!(p, en_list, label=label, linewidth=2)
    return p
end

function make_grid(c₀)
    # make initial zero grid
    c = zeros(N,N)
    # set 0.5+(1.0+c₀) of upper as 1.0
    i = round(Int, 0.5*(1.0+c₀)*N)
    c[1:i,:] .= 1.0
    # set the rest to -1.0
    c[i+1:end,:] .= -1.0
    c .+= c₀ - mean(c)
    # shift matrix to center
    c = circshift(c, (-N÷4, -N÷4))
    return c
end

# make initial uniform grid
#c = make_grid(c₀)
#c = zeros(N,N)
c = ones(N,N).*c₀
#c = load_state("cwt.jld2")

# relaxation
#c_list = solver_noise(c, 100000, save_every=1000, plot_every=1000, noisy=false)
#c = c_list[end]
# check if energy makes sense
println("Energy = ", energy(c), " expected = ", 2*γ*L)
# run noise
c_list_noise = solver_noise(c, round(Int, 500.0/Δt), save_every=round(Int, 0.1/Δt), plot_every=10000, noisy=true)
#c_list_noise = solver_noise(c, round(Int, 10.0/Δt), save_every=round(Int, 0.01/Δt), plot_every=1000, noisy=true)
# plot energy kernel density for 2nd half
p1 = plot_energy_kernel_density(c_list_noise[length(c_list_noise)÷2:end], "Noise")
p2 = plot_c_kernel_density(c_list_noise[length(c_list_noise)÷2:end], "Noise")
# run MC
c_list_mc = solver_mc(c_list_noise[end-1], 50000, save_every=100, plot_every=5000)
# plot energy kernel density
p = plot_energy_kernel_density(c_list_mc[length(c_list_mc)÷2:end], "MC", p=p1)
p = plot_c_kernel_density(c_list_mc[length(c_list_mc)÷2:end], "MC", p=p2)
#p = plot_energy_kernel_density(c_list_noise[10:end], "Noise")
# save figure
savefig(p1, "energy_kernel_density.png")
savefig(p2, "c_kernel_density.png")