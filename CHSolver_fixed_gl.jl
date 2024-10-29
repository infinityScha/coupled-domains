using Statistics
using LinearAlgebra
using Interpolations
using KernelDensity
using FFTW
using Roots
using Plots
using DataFrames
using CSV
using ProgressMeter
using Printf
using JLD2
using ColorSchemes
using Colors

# set GR plotting backend
ENV["GKSwstype"] = "100"

# Semi-implicit Fourier spectral method for the Cahn-Hilliard equation 
# following https://www.sciencedirect.com/science/article/pii/S001046559700115X

which_lipid = ARGS[1]

γₑₗs = Dict("DPPC"=>0.25, "DAPC"=>0.7)

# physical parameters
const I = 2/3*sqrt(2) #2/3*sqrt(2) #0.85714⋅√2 for f(c)=0.25*(c⁶-1)², for regular CH it is 2⋅√2/3
const γₗᵢₚ = parse(Float64, ARGS[2])
const σ = 1.2 # interface width [nm] = √(K/2ϵ)
const ζ = 1.0 # Cook noise should be normalized to kT, in this case kT = 1
const λ = parse(Float64, ARGS[3]) # interleaflet coupling [kT/nm²]
const dλdt = 0.000 # rate of change of interleaflet coupling [kT/nm²/ns]
const max_noise_wavelength = 2.5 # [nm], 5 nm should give a good thermal noise
const Kᴸᵒ = 23.0 # bending modulus of Lo phase [kT]
const Kᴸᵈ = 9.0 # bending modulus of Ld phase [kT]
const γₑₗ = γₑₗs[which_lipid]
println(γₑₗ)
const γ = γₑₗ + γₗᵢₚ # line tension [kT/nm] = √I(K⋅ϵ)
const R = parse(Float64, ARGS[4]) # radius of curvature [nm]
const Jₛᴸᵒ = -0.30 # spontaneous curvature of Lo phase [nm^-1]
const Jₛᴸᵈ = -0.30 # spontaneous curvature of Ld phase [nm^-1]
const c₀ = parse(Float64, ARGS[5]) # bulk field value 
# numerical parameters
const M₁ = 7 # log(2, grid size)
const M₂ = 6 # log(2, grid size)
const L₁ = 32.0 # box size [nm]
const Δt = 0.05 # time step [ns]
# deduced numerical & physical parameters
const N₁ = 2^M₁ # grid size
const N₂ = 2^M₂ # grid size
const K = sqrt(2)*γ*σ/I
const ϵ = γ/(sqrt(2)*I*σ)
const L₂ = 14.0 # box size [nm]

const γₗ = 2*γₑₗ/I*sqrt(K/ϵ) # indirect coupling [kT]
println("K = ", K, " ϵ = ", ϵ, " γ = ", γ, " γₗ = ", γₗ)

# external field matrix
const H₁ˡᵒ = zeros(N₁,N₂)
const H₁ˡᵈ = zeros(N₁,N₂)
const H₂ˡᵒ = zeros(N₁,N₂)
const H₂ˡᵈ = zeros(N₁,N₂)
# approximate curvature matrix
function Hij(i, R, is_lo=true)
    if abs(R) > 1000.0
        return 0.0
    end
    c = sin(2*π*i/(N₁+1))/R
    if is_lo
        return 0.5*Kᴸᵒ*(c + Jₛᴸᵒ)^2 # at Φ = 1
    end
    return 0.5*Kᴸᵈ*(c + Jₛᴸᵈ)^2 # at Φ = -1
end

for i in 1:N₁
    for j in 1:N₂
        H₁ˡᵒ[i,j] = Hij(i, R)
        H₁ˡᵈ[i,j] = Hij(i, R, false)
        H₂ˡᵒ[i,j] = Hij(i, -R)
        H₂ˡᵈ[i,j] = Hij(i, -R, false)
    end
end

const H₁ = 0.5*(H₁ˡᵒ .- H₁ˡᵈ)
const H₂ = 0.5*(H₂ˡᵒ .- H₂ˡᵈ)

# deduced Fourier parameters
const k₁ = 2.0*π/L₁*cat(0:N₁/2-1, -N₁/2:-1, dims=1)
const k₂ = 2.0*π/L₂*cat(0:N₂/2-1, -N₂/2:-1, dims=1)
const k² = zeros(N₁,N₂)
for i in 1:N₁
    for j in 1:N₂
        k²[i,j] = k₁[i]^2 + k₂[j]^2
    end
end
const k⁴ = k².^2
const k_vecs = zeros(N₁,N₂,2)
for i in 1:N₁
    for j in 1:N₂
        k_vecs[i,j,1] = k₁[i]
        k_vecs[i,j,2] = k₂[j]
    end
end
# deduced semi-implicit parameters
temp = 1.0 .+ K*Δt*k⁴ 
const A = 1.0./temp
const B = -ϵ*Δt*k²./temp
const C = -Δt./temp
const η = sqrt(2.0*ζ/Δt/L₁*N₁/L₂*N₂)
const mask = k² .> (1.001*4.0*π*π/max_noise_wavelength/max_noise_wavelength)
# filename and folder for saving
# filename include the ARGS
const folder = "new_model/$(ARGS[1])/$(ARGS[2])/$(ARGS[3])/$(ARGS[4])/$(ARGS[5])/$(ARGS[6])/"
const filename = ARGS[7]
# make folder if it does not exist
if !isdir(folder)
    mkpath(folder)
end

function colorfunc(c)
    return clamp(c, -1.0, 1.0)
end

function plot_spectrum(cᵤ, cₗ, coupling_list, fit_list, ;step=1.0)
    cᵤ_colors = colorfunc.(cᵤ)
    cₗ_colors = colorfunc.(cₗ)
    # plot on p
    heatmap1 = heatmap(cᵤ_colors, aspect_ratio=1, color=:bluesreds, cbar=false, clim=(-1,1), axis=nothing, border=:none, ticks=nothing, title = "Upper leaflet")
    heatmap2 = heatmap(cₗ_colors, aspect_ratio=1, color=:bluesreds, cbar=false, clim=(-1,1), axis=nothing, border=:none, ticks=nothing, title = "Lower leaflet")
    # plot on the curvature on the heatmaps on the y axis
    plot!(heatmap1, 0.5*N₂.-sin.(2*π*range(0.0, L₁, N₁)/L₁)*3.0, range(1.0, N₁, N₁), lw=2, color=:black, legend=false)
    plot!(heatmap2, 0.5*N₂.-sin.(2*π*range(0.0, L₁, N₁)/L₁)*3.0, range(1.0, N₁, N₁), lw=2, color=:black, legend=false)
    t = LinRange(0, step*Δt*length(coupling_list), length(coupling_list))
    coupling_plot = plot(t, coupling_list, xlims = (t[1],t[end]), ylims=(0,1.0), color=:black, label="Coupling ratio")
    fitᵤ = [x[1] for x in fit_list]
    fitₗ = [x[2] for x in fit_list]
    plot!(coupling_plot, t, fitᵤ, color=:red, label="fit u")
    plot!(coupling_plot, t, fitₗ, color=:blue, label="fit l")
    # plot the heatmaps and the coupling plot
    p = plot(heatmap1, heatmap2, coupling_plot, layout=(1,3), size=(1450,500))
    # tight layout
    plot!(framestyle=:none, foreground_color=:black)
    return p
end

function plot_c_distribution(cᵤ, cₗ)
    # kdes
    k1 = kde(cᵤ[:])
    k2 = kde(cₗ[:])
    p1 = plot(k1.x, k1.density, color=:red, label="u")
    p2 = plot(k2.x, k2.density, color=:blue, label="l")
    p = plot(p1, p2, layout=(1,2), size=(1000,500))
    savefig(folder*filename*"_dist.png")
end

function _center_point(c)
    # find the center point of the c distribution
    # average c over the 1st dimension
    c = 0.5.*(mean(c, dims=2)[:] .+ 1.0)
    # find first fourier mode
    w = exp.(-2im * π * collect(0:N₁-1) / N₁)
    f1 = dot(c, w)
    angle = Base.angle(conj(f1))
    if angle < 0
        angle += 2π
    end
    approx_center = L₁ * angle / (2π)
    x = range(0.0, L₁, N₁+1)[1:end-1]
    x_shifted = mod1.(x .- approx_center .+ L₁/2, L₁)
    # find the center point via zero point of cumsum
    argsort_x_shifted = sortperm(x_shifted)
    x_shifted = x_shifted[argsort_x_shifted]
    c_shifted = c[argsort_x_shifted]
    cumsum_c = cumsum(c_shifted)
    cumsum_c /= cumsum_c[end] + cumsum_c[1]
    # do linear interpolation of cumsum
    f = interpolate((x_shifted,), cumsum_c .- 0.5, Gridded(Linear()))
    f = extrapolate(f, Flat())
    # find the zero point, try catch for errors
    center = L₁/2
    try
        center = mod1(find_zero(f, (0.0, L₁), Bisection()) - L₁/2 + approx_center, L₁)
    catch e
        # plot cumsum_c
        p = plot(x_shifted, cumsum_c, label="cumsum_c")
        plot!(p, x_shifted, f(x_shifted), label="f")
        savefig("cumsum_c.png")
    end
    return center 
end

function plot_1d_averaged_profile(c_list, ld_com=false)
    cᵤs = [c[1] for c in c_list]
    cₗs = [c[2] for c in c_list]
    # keep only the second half of the profiles
    if size(cᵤs)[1] > 1
        cᵤs = cᵤs[round(Int, length(cᵤs)/2):end]
        cₗs = cₗs[round(Int, length(cₗs)/2):end]
    end
    # fit ld com to be at 0 if needed
    if ld_com
        peaks = [_center_point(c) for c in cᵤs]
        xs = [range(0.0, L₁, N₁+1)[1:end-1] for c in cᵤs]
        xs_shifted = [mod1.((x .- peak .+ L₁/2), L₁) for (x, peak) in zip(xs, peaks)]
        xs_shifted_argsort = [sortperm(x) for x in xs_shifted]
        cᵤs_interp = [interpolate((x[argsort_x],), mean(c[argsort_x], dims=2)[:], Gridded(Linear(Periodic()))) for (x, c, argsort_x) in zip(xs_shifted, cᵤs, xs_shifted_argsort)]
        cₗs_interp = [interpolate((x[argsort_x],), mean(c[argsort_x], dims=2)[:], Gridded(Linear(Periodic()))) for (x, c, argsort_x) in zip(xs_shifted, cₗs, xs_shifted_argsort)]
        cᵤs_extrap = [extrapolate(c, Periodic()) for c in cᵤs_interp]
        cₗs_extrap = [extrapolate(c, Periodic()) for c in cₗs_interp]
        cᵤs = [c(x) for (c, x) in zip(cᵤs_extrap, xs)]
        cₗs = [c(x) for (c, x) in zip(cₗs_extrap, xs)]
    end
    # average over all cᵤs and cₗs
    cᵤ = reduce(+, cᵤs) / length(cᵤs)
    cₗ = reduce(+, cₗs) / length(cₗs)
    x = range(0.0, L₁, N₁+1)[1:end-1]
    p = plot(x, cᵤ, color=:red, label="u", legend=false, title="1D averaged profile", xlabel="x [nm]", ylabel="c", size=(1000,500), xlims=(0,L₁), ylims=(-1,1), xticks=0:1:L₁)
    plot!(x, cₗ, color=:blue, label="l")
    savefig(folder*filename*"_1d_profile.png")
    # save to CSV as well
    CSV.write(folder*filename*"_1d_profile.csv", DataFrame(x=x, cᵤ=cᵤ, cₗ=cₗ))
end

function coupling_ratio(cᵤ, cₗ)
    agreement = (cᵤ .* cₗ) .> 0.0
    return mean(agreement)
    #return 1.0 - 0.5*mean(abs.(cᵤ .- cₗ))
end

function domain_fit(cᵤ, cₗ)
    fitᵤ = cᵤ.*(H₁.-mean(H₁)) .< 0.0
    fitₗ = cₗ.*(H₂.-mean(H₂)) .< 0.0
    return mean(fitᵤ), mean(fitₗ)
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

function save_c_list(c_list, filename="CHSolver.jld2")
    # save c_list
    save(filename, "c_list", c_list)
end

function load_c_list(filename="CHSolver.jld2")
    # load c_list
    c_list = load(filename, "c_list")
    return c_list
end

function divergence(v₁, v₂)
    # compute divergence of vector field
    ∂₁v₁, ∂₂v₂ = zeros(N₁,N₂), zeros(N₁,N₂)
    ∂₁v₁ = real(ifft(im*k_vecs[:,:,1].*fft(v₁)))
    ∂₂v₂ = real(ifft(im*k_vecs[:,:,2].*fft(v₂)))
    return (∂₁v₁ .+ ∂₂v₂)
end

function noise()
    f₁ = randn(N₁, N₂)
    f₂ = randn(N₁, N₂)
    # compute divergence of random vector field
    f̂ = fft(divergence(f₁, f₂)*η)
    # set zero mode to zero (no global translation, theoretically should be already zero but numerical errors...)
    f̂[1,1] = 0.0
    # keep only fluctuations with k > max_noise_wavelength^-1 
    f̂[mask] .*= 0.0
    return f̂
end

function solver(cᵤ, cₗ, Nsteps, ;save_every=100, plot_every=0, initial_λ=λ, noisy = true, verbose = true)
    λ_ = initial_λ
    c_list = []#[[cᵤ, cₗ]]
    coupling_list_save = [coupling_ratio(cᵤ, cₗ)]
    coupling_list_plot = [coupling_list_save[end]]
    fit_list_save = [domain_fit(cᵤ, cₗ)]
    fit_list_plot = [fit_list_save[end]]
    # only make progress bar if verbose
    pbar = verbose ? Progress(Nsteps, barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
    barlen=10) : nothing
    for i in 1:Nsteps
        # Fourier transform
        ĉⁿᵤ = fft(cᵤ)
        ĉⁿₗ = fft(cₗ)
        # random noise
        f̂ᵤ, f̂ₗ = noisy ? (noise(), noise()) : (zeros(N₁,N₂), zeros(N₁,N₂))
        d2cᵤ = real(ifft(-k².*ĉⁿᵤ))
        d2cₗ = real(ifft(-k².*ĉⁿₗ))
        ĉⁿ⁺¹ᵤ = A.*ĉⁿᵤ .+ B.*fft(cᵤ.^3 - cᵤ .- 0.25*λ_.*(cₗ .- cᵤ) .- 0.5*γₗ.*d2cₗ .+ H₁) + C.*f̂ᵤ
        ĉⁿ⁺¹ₗ = A.*ĉⁿₗ .+ B.*fft(cₗ.^3 - cₗ .- 0.25*λ_.*(cᵤ .- cₗ) .- 0.5*γₗ.*d2cᵤ .+ H₂) + C.*f̂ₗ
        # more BS
        #cᵤ⁵ = cᵤ.^5
        #cₗ⁵ = cₗ.^5
        # compute nonlinear term
        #ĉⁿ⁺¹ᵤ = A.*ĉⁿᵤ .+ B.*fft(3.0.*cᵤ⁵.*(cᵤ⁵.*cᵤ .- 1.0) - 0.25*λ_.*(cₗ .- cᵤ) .- 0.5*γₗ.*d2cₗ .+ H₁) + C.*f̂ᵤ   
        #ĉⁿ⁺¹ₗ = A.*ĉⁿₗ .+ B.*fft(3.0.*cₗ⁵.*(cₗ⁵.*cₗ .- 1.0) - 0.25*λ_.*(cᵤ .- cₗ) .- 0.5*γₗ.*d2cᵤ .+ H₂) + C.*f̂ₗ
        #cᵤ³ = cᵤ.^3
        #cₗ³ = cₗ.^3
        #ĉⁿ⁺¹ᵤ = A.*ĉⁿᵤ .+ B.*fft(2.0*cᵤ³.*(cᵤ³.*cᵤ .- 1.0) - 0.25*λ_.*(cₗ .- cᵤ) .+ H₁) + C.*f̂ᵤ
        #ĉⁿ⁺¹ₗ = A.*ĉⁿₗ .+ B.*fft(2.0*cₗ³.*(cₗ³.*cₗ .- 1.0) - 0.25*λ_.*(cᵤ .- cₗ) .+ H₂) + C.*f̂ₗ
        # inverse Fourier transform
        cᵤ = real(ifft(ĉⁿ⁺¹ᵤ))
        cₗ = real(ifft(ĉⁿ⁺¹ₗ))
        # save
        if (save_every != 0) && (i % save_every == 0) && (i*Δt > 100.0)
            push!(coupling_list_save, coupling_ratio(cᵤ, cₗ))
            push!(fit_list_save, domain_fit(cᵤ, cₗ))
            push!(c_list, [cᵤ, cₗ])
            # set progressbar title to coupling ratio
            if verbose
                pbar.desc = @sprintf("%.3f %.3f %.3f %.3f | Time: %.3f", coupling_list_save[end], λ_, maximum(cᵤ), minimum(cᵤ), i*Δt)
            end
        end
        if (plot_every != 0) && (i % plot_every == 0) #&& (i*Δt > 100.0)
            push!(coupling_list_plot, coupling_ratio(cᵤ, cₗ))
            push!(fit_list_plot, domain_fit(cᵤ, cₗ))
            p = plot_spectrum(cᵤ, cₗ, coupling_list_plot, fit_list_plot, step = plot_every)
            savefig("curr.png")
            #plot_c_distribution(cᵤ, cₗ)
            #plot_1d_averaged_profile(c_list, true)
            if verbose
                pbar.desc = @sprintf("%.3f %.3f %.3f %.3f | Time: %.3f", coupling_list_plot[end], λ_, maximum(cᵤ), minimum(cᵤ), i*Δt)
            end
        end
        if verbose
            next!(pbar)
        end
        λ_ = max(λ_ + dλdt * Δt, 0.0)
    end
    plot_1d_averaged_profile(c_list, true)
    return c_list, coupling_list_save, fit_list_save
end

function animate_time_evolution(c_list, coupling_list, fit_list, filename="CHSolver.gif")
    anim = @animate for i in eachindex(c_list)
        p = plot_spectrum(c_list[i][1], c_list[i][2], coupling_list[1:i], fit_list[1:i])
    end
    gif(anim, filename, fps=10)
end

function plot_coupling_list(coupling_list)
    plot(coupling_list, xlims=(1, length(coupling_list)), ylims=(0,1.0), color=:black, legend=false, title="Coupling ratio")
    savefig("coupling_list.png")
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

function energy(cᵤ, cₗ)
    #TODO: include γₗ
    # compute energy
    #cᵤ⁶ = cᵤ.^6 
    #cₗ⁶ = cₗ.^6
    #cᵤ⁴ = cᵤ.^4
    #cₗ⁴ = cₗ.^4
    diff_c₁ᵤ, diff_c₂ᵤ = diff(cᵤ)
    diff_c₁ₗ, diff_c₂ₗ = diff(cₗ)
    #enᵤ = (ϵ*0.25*sum((cᵤ⁶ .- 1.0).^2) + 0.5*K*sum(diff_c₁ᵤ.^2 + diff_c₂ᵤ.^2) + 0.125*λ*sum((cᵤ .- cₗ).^2) + 0.5 * sum((cᵤ.+1.0).*H₁ˡᵒ - (cᵤ .- 1.0).*H₁ˡᵈ))*(L/N)^2 
    #enₗ = (ϵ*0.25*sum((cₗ⁶ .- 1.0).^2) + 0.5*K*sum(diff_c₁ₗ.^2 + diff_c₂ₗ.^2) + 0.125*λ*sum((cₗ .- cᵤ).^2) + 0.5 * sum((cₗ.+1.0).*H₂ˡᵒ - (cₗ .- 1.0).*H₂ˡᵈ))*(L/N)^2
    #enᵤ = (ϵ*0.25*sum((cᵤ⁴ .- 1.0).^2) + 0.5*K*sum(diff_c₁ᵤ.^2 + diff_c₂ᵤ.^2) + 0.125*λ*sum((cᵤ .- cₗ).^2) + 0.5 * sum((cᵤ.+1.0).*H₁ˡᵒ - (cᵤ .- 1.0).*H₁ˡᵈ))*(L/N)^2
    #enₗ = (ϵ*0.25*sum((cₗ⁴ .- 1.0).^2) + 0.5*K*sum(diff_c₁ₗ.^2 + diff_c₂ₗ.^2) + 0.125*λ*sum((cₗ .- cᵤ).^2) + 0.5 * sum((cₗ.+1.0).*H₂ˡᵒ - (cₗ .- 1.0).*H₂ˡᵈ))*(L/N)^2
    enᵤ = (ϵ*0.25*sum((cᵤ.^2 .- 1.0).^2) + 0.5*K*sum(diff_c₁ᵤ.^2 + diff_c₂ᵤ.^2) + 0.125*λ*sum((cᵤ .- cₗ).^2) + 0.5 * sum((cᵤ.+1.0).*H₁ˡᵒ - (cᵤ .- 1.0).*H₁ˡᵈ))*(L₁/N₁)*(L₂/N₂)
    enₗ = (ϵ*0.25*sum((cₗ.^2 .- 1.0).^2) + 0.5*K*sum(diff_c₁ₗ.^2 + diff_c₂ₗ.^2) + 0.125*λ*sum((cₗ .- cᵤ).^2) + 0.5 * sum((cₗ.+1.0).*H₂ˡᵒ - (cₗ .- 1.0).*H₂ˡᵈ))*(L₁/N₁)*(L₂/N₂)
    return enᵤ + enₗ
end


function make_grid(c₀,; upper=true, to_mid=false)
    # make initial zero grid
    c = zeros(N₁,N₂)
    if to_mid
        i = round(Int, 0.5*(1.0+c₀)*N₁)
        c[1:i,:] .= 1.0
        # set the rest to -1.0
        c[i+1:end,:] .= -1.0
        # shift by N/4
        c = circshift(c, (N₁÷4,0))
        c .+= c₀ - mean(c)
        return c
    end
    if upper
        # set 0.5+(1.0+c₀) of upper as 1.0
        i = round(Int, 0.5*(1.0+c₀)*N₁)
        c[1:i,:] .= 1.0
        # set the rest to -1.0
        c[i+1:end,:] .= -1.0
    else
        # set 0.5+(1.0-c₀) of upper as -1.0
        i = round(Int, 0.5*(1.0-c₀)*N₁)
        c[1:i,:] .= -1.0
        # set the rest to 1.0
        c[i+1:end,:] .= 1.0
    end
    c .+= c₀ - mean(c)
    return c
end

# make initial uniform grid
is_uniform = parse(Bool, ARGS[6])
if is_uniform
    cᵤ = zeros(N₁,N₂) .+ c₀
    cₗ = zeros(N₁,N₂) .+ c₀
else 
    cᵤ = make_grid(c₀, to_mid=true)
    cₗ = make_grid(c₀, to_mid=true)

end

# solve
c_list, coupling_list, fit_list = solver(cᵤ, cₗ, round(Int, 5000/Δt), save_every=round(Int, 50/Δt), plot_every=0, verbose=false)
# save c_list
save_c_list(c_list[end-19,end], folder*filename*".jld2")