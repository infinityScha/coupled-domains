using Statistics
using Plots
using ProgressMeter
using FFTW
using Printf
using JLD2
using ColorSchemes
using Colors

# set GR plotting backend
gr()

# Semi-implicit Fourier spectral method for the Cahn-Hilliard equation 
# following https://www.sciencedirect.com/science/article/pii/S001046559700115X

# physical parameters
const I = 0.85714*sqrt(2) # for f(c)=0.25*(c⁶-1)², for regular CH it is 2/3*sqrt(2)
const γ = 3.0 # line tension [kT/nm] = I√(K⋅ϵ)
const σ = 1.0 # interface width [nm] = sqrt(K/ϵ)
const ζ = 1.0e-1 # Cook noise [kT/ns]
const c₀ = 0.0 # bulk field value
# numerical parameters
const M = 9 # log(2, grid size)
const L = 40.0 # box size [nm]
const Δt = 1.0e-3 # time step [ns]
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

# same for 1D line 
kₗ = (2.0*π/L)*cat(1:N/2-1, dims=1)
kₗ² = kₗ.^2

# deduced semi-implicit parameters
temp = 1.0 .+ K*Δt*k⁴
const A = 1.0./temp
const B = -ϵ*Δt*k²./temp
const C = -Δt./temp
const η = sqrt(2.0*ζ/Δt)/L*N

println("K = ", K, " [kT]", ", ϵ = ", ϵ, " [kT/nm^2]")

# Define a custom color map with a gradient of 5 colors
const mycolors = range(RGB(0.4,0.6,1), stop=RGB(1,0.6,0.4), length=5)

# Create a function to map values of c to colors
function colorfunc(c)
    if c < -0.3
        return mycolors[1]
    elseif c < -0.1
        return mycolors[2]
    elseif c < 0.1
        return RGB(1,0,0)
    elseif c < 0.3
        return mycolors[4]
    else
        return mycolors[5]
    end
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
    ∂₁v₁ = zeros(N,N)
    ∂₂v₂ = zeros(N,N)
    for i in 1:N
        for j in 1:N
            # 2nd order approximate
            ∂₁v₁[i,j] = (v₁[mod1(i+1,N),j] - v₁[mod1(i-1,N),j])/2.0
            ∂₂v₂[i,j] = (v₂[i,mod1(j+1,N)] - v₂[i,mod1(j-1,N)])/2.0
        end
    end
    return (∂₁v₁ .+ ∂₂v₂)./L*N
end

function noise()
    f₁ = randn(N, N)
    f₂ = randn(N, N)
    # compute divergence of random vector field
    f̂ = fft(divergence(f₁, f₂)*η)
    # keep only fluctuations with k > nm -1
    f̂[k² .> (4.0*π*π)] .= 0.0
    return f̂
end

function solver(c, Nsteps, ;save_every=100, plot_every=0)
    c_list = [c]
    pbar = Progress(Nsteps)
    for i in 1:Nsteps
        # Fourier transform
        ĉⁿ = fft(c)
        # random noise
        f̂ = noise()
        # compute nonlinear term
        c⁵ = c.^5
        ĉⁿ⁺¹ = A.*ĉⁿ .+ B.*fft(3.0.*c⁵.*(c⁵.*c .- 1.0)) + C.*f̂
        #ĉⁿ⁺¹ = A.*(ĉⁿ .+ f̂) .+ B.*fft(-c .+ c.^3) 
        # inverse Fourier transform
        c = real(ifft(ĉⁿ⁺¹))
        # save
        if (save_every != 0) && (i % save_every == 0)
            push!(c_list, c)
        end
        if (plot_every != 0) && (i % plot_every == 0)
            p = plot_spectrum(c)
            display(p)
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

# Calculate line tension from capillary wave theory
function find_line_spectrum(c)
    # find the row with the larges mean value of c
    iₘₐₓ = argmin(mean(c, dims=2))
    iₘₐₓ = iₘₐₓ[1]
    ĵ₀_arr = []
    # do for the two interfaces
    for i in 1:2
        # find for each column where c is closest to 0
        j₀ = zeros(Float64, N)
        for j in 1:N
            if i == 1
                j₀[j] = Float64(argmin(abs.(c[1:iₘₐₓ, j])))
            else
                j₀[j] = Float64(argmin(abs.(c[iₘₐₓ:end, j])))
            end
        end
        j₀ .*= L/N
        j₀ .-= mean(j₀)
        #display(plot(j₀))
        # pause for 2 seconds
        #sleep(2)
        # compute fourier series of j₀
        ĵ₀ = fft(j₀)./N
        # get only the positive frequencies
        ĵ₀ = ĵ₀[2:Int64(N/2)]
        # absulute value squared multiplied by kₗ²
        ĵ₀ = abs2.(ĵ₀).*kₗ².*L
        # push to array
        push!(ĵ₀_arr, ĵ₀)
    end
    # compute mean of the two interfaces
    return ĵ₀_arr
end

function cwt_helper(ĵ_list)
    j̄ = zeros(Int64(N/2-1))
    for ĵ in ĵ_list
        j̄ .+= ĵ
    end
    j̄ ./= length(ĵ_list)
    j̄⁻¹ = 1.0./j̄
    return j̄⁻¹
end

function cwt(c_list)
    ĵ_list = []
    for c in c_list
        ĵ₁, ĵ₂ = find_line_spectrum(c)
        push!(ĵ_list, ĵ₁)
        push!(ĵ_list, ĵ₂)
    end
    # make folder for histograms if it doesn't exist
    if !isdir("cwt")
        mkdir("cwt")
    end
    # remove all files in folder
    for file in readdir("cwt")
        rm("cwt/$file")
    end
    # plot histogram for each frequency
    for i in 1:Int64(N/2-1)
        ĵ = [ĵ_[i] for ĵ_ in ĵ_list]
        histogram(ĵ, bins=range(0.0, 0.1, 30), legend=false)
        # save as figure
        savefig("cwt/$(i).png")
    end
    # compute mean for each frequency
    j̄⁻¹ = cwt_helper(ĵ_list)
    # estimate STE by splitting ĵ_list to 10 parts and doing the same as above
    j̄_sem = []
    for i in 1:10
        push!(j̄_sem, cwt_helper(ĵ_list[Int64(round(Int,length(ĵ_list)/10*(i-1)))+1:Int64(round(Int, length(ĵ_list)/10*i))]))
    end
    # get STE from STD
    j̄_sem = std(j̄_sem, dims=1)/sqrt(10)
    
    # plot predicted γ vs kₗ (with error bars)
    p = plot(kₗ[kₗ .< (2.0*π)]./(2.0*π), j̄⁻¹[kₗ .< (2.0*π)], yerr=j̄_sem[kₗ .< (2.0*π)], color=:black, legend=false, xlabel="k [1 / nm]", ylabel="gamma [kT / nm]", title="gamma vs k")
    # plot line for real y=γ for comparison
    plot!(p, kₗ[kₗ .< (2.0*π)]./(2.0*π), γ*ones(sum(kₗ .< (2.0*π))), color=:black, linestyle=:dash)
    savefig("cwt.png")
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
c = make_grid(c₀)
#c = load_state("cwt.jld2")

# solve
#c_list = solver(c, 100000, save_every=25, plot_every=50)
#save_state(c_list[end], "cwt.jld2")
#save_state_list(c_list, "cwt_list.jld2")
c_list = load_state_list("cwt_list.jld2")
#c_list2 = solver(c_list[end], 25000, save_every=50, plot_every=50)
#c_list = append!(c_list, c_list2)
#save_state_list(c_list, "cwt_list.jld2")
# plot line tension
cwt(c_list[250:end])
#animate_time_evolution(c_list, coupling_list, "cwt.gif")