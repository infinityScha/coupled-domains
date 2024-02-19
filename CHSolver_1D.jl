using Statistics
using Plots
using ProgressMeter
using FFTW
using Printf
using JLD2
using ColorSchemes
using Colors

# set GR plotting backend
ENV["GKSwstype"] = "100"
#gr()


# Semi-implicit Fourier spectral method for the Cahn-Hilliard equation 
# following https://www.sciencedirect.com/science/article/pii/S001046559700115X

# physical parameters
const I = 2/3*sqrt(2) #0.85714*sqrt(2) #0.8*sqrt(2) #0.85714*sqrt(2) # for f(c)=0.25*(c⁶-1)², for regular CH it is 2⋅√2/3
const γ = 3.0 # line tension [kT/nm] = I√(K⋅ϵ)
const σ = 0.7 # interface width [nm] = √(K/ϵ)
const ζ = 1.0 # Cook noise [ns⋅kT/nm] # TODO: check units
const λ = 0.3 #0.16 #0.33 # interleaflet coupling [kT/nm²]
const dλdt = 0.0000 # rate of change of interleaflet coupling [kT/nm²/ns]
const max_noise_wavelength = 0.0 # [nm]
const Kᴸᵒ = 22.9 # bending modulus of Lo phase [kT]
const Kᴸᵈ = 8.9 # bending modulus of Ld phase [kT]
const R = 10.0#10.0 # radius of curvature at the maximum [nm] 
const Jₛᴸᵒ = -0.178 # spontaneous curvature of Lo phase [nm^-1]
const Jₛᴸᵈ = 0.114 # spontaneous curvature of Ld phase [nm^-1]
const m = 1.0 # mobility [nm^2/ns/kT]
const c₀ = 0.0 # bulk field value
# numerical parameters
const M = 5 # log(2, grid size)
const L = 20.0 #max_noise_wavelength * (2^M) # box size [nm]
const Δt = 1.0e-2 # time step [ns]
# deduced numerical & physical parameters
const N = 2^M # grid size
const K = γ*σ/I
const ϵ = K/σ^2

# external field matrix
const H₁ˡᵒ = zeros(N)
const H₁ˡᵈ = zeros(N)
const H₂ˡᵒ = zeros(N)
const H₂ˡᵈ = zeros(N)
# approximate curvature matrix
function Hij(i, R, is_lo=true)
    c = 0.5*sin(2*π*i/N)/R
    if is_lo
        return 0.5*Kᴸᵒ*(c + Jₛᴸᵒ)^2 # at Φ = 1
    end
    return 0.5*Kᴸᵈ*(c + Jₛᴸᵈ)^2 # at Φ = -1
end

for i in 1:N
    H₁ˡᵒ[i] = Hij(i, R)
    H₁ˡᵈ[i] = Hij(i, R, false)
    H₂ˡᵒ[i] = Hij(i, -R)
    H₂ˡᵈ[i] = Hij(i, -R, false)
end

const H₁ = 0.5*(H₁ˡᵒ .- H₁ˡᵈ)
const H₂ = 0.5*(H₂ˡᵒ .- H₂ˡᵈ)

# deduced Fourier parameters
const k = 2.0*π/L*cat(0:N/2-1, -N/2:-1, dims=1)
const k² = k.^2
const k⁴ = k².^2
# deduced semi-implicit parameters
temp = 1.0 .+ K*Δt*k⁴
const A = 1.0./temp
const B = -ϵ*Δt*k²./temp
const C = -Δt./temp
const η = sqrt(2.0*ζ/Δt*N)/L

function colorfunc(c)
    return clamp(c, -1.0, 1.0)
end

function plot_spectrum(cᵤ, cₗ, coupling_list, fit_list, ; step = 1.0)
    # plot on p
    cᵤ, cₗ = clamp.(cᵤ, -1.0, 1.0), clamp.(cₗ, -1.0, 1.0)
    plot1 = plot(cᵤ, color=:blue)
    plot!(plot1, cₗ, color=:red)
    t = LinRange(0, step*Δt*length(coupling_list), length(coupling_list))
    coupling_plot = plot(t, coupling_list, xlims = (t[1],t[end]), ylims=(0,1.0), color=:black, label="Coupling ratio")
    fitᵤ = [x[1] for x in fit_list]
    fitₗ = [x[2] for x in fit_list]
    plot!(coupling_plot, t, fitᵤ, color=:red, label="fit u")
    plot!(coupling_plot, t, fitₗ, color=:blue, label="fit l")
    p = plot(plot1, coupling_plot, layout=(1,2), size=(1000,500))
    plot!(framestyle=:box, foreground_color=:black)
    return p
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

function divergence(v)
    # compute divergence of vector field
    ∂v = real(ifft(im*k.*fft(v)))
    return ∂v
end

function noise()
    f = randn(N)
    # compute divergence of random vector field
    f̂ = fft(divergence(f)*η)
    f̂[1] = 0.0
    # keep only fluctuations with k > max_noise_wavelength^-1 
    mask = k² .> (1.001*4.0*π*π/max_noise_wavelength/max_noise_wavelength)
    f̂[mask] .*= 0.0 #4.0*π*π/max_noise_wavelength/max_noise_wavelength./k²[mask]
    return f̂
end

function solver(cᵤ, cₗ, Nsteps, ;save_every=100, plot_every=0, initial_λ=λ, noisy = true, verbose = true)
    λ_ = initial_λ
    c_list = [[cᵤ, cₗ]]
    coupling_list_save = [coupling_ratio(cᵤ, cₗ)]
    coupling_list_plot = [coupling_list_save[end]]
    fit_list_save = [domain_fit(cᵤ, cₗ)]
    fit_list_plot = [fit_list_save[end]]
    # only make progress bar if verbose
    pbar = verbose ? Progress(Nsteps) : nothing
    for i in 1:Nsteps
        # Fourier transform
        ĉⁿᵤ = fft(cᵤ)
        ĉⁿₗ = fft(cₗ)
        # random noise
        f̂ᵤ, f̂ₗ = noisy ? (noise(), noise()) : (zeros(N), zeros(N))
        # more BS
        #cᵤ⁵ = cᵤ.^5
        #cₗ⁵ = cₗ.^5
        # compute nonlinear term
        #ĉⁿ⁺¹ᵤ = A.*ĉⁿᵤ .+ B.*fft(3.0.*cᵤ⁵.*(cᵤ⁵.*cᵤ .- 1.0) - 0.25*λ_.*(cₗ .- cᵤ) .+ H₁) + C.*f̂ᵤ   
        #ĉⁿ⁺¹ₗ = A.*ĉⁿₗ .+ B.*fft(3.0.*cₗ⁵.*(cₗ⁵.*cₗ .- 1.0) - 0.25*λ_.*(cᵤ .- cₗ) .+ H₂) + C.*f̂ₗ
        ĉⁿ⁺¹ᵤ = A.*ĉⁿᵤ .+ B.*fft(cᵤ.^3 - cᵤ .- 0.25*λ_.*(cₗ .- cᵤ) .+ H₁) + C.*f̂ᵤ
        ĉⁿ⁺¹ₗ = A.*ĉⁿₗ .+ B.*fft(cₗ.^3 - cₗ .- 0.25*λ_.*(cᵤ .- cₗ) .+ H₂) + C.*f̂ₗ
        #cᵤ³ = cᵤ.^3
        #cₗ³ = cₗ.^3
        #ĉⁿ⁺¹ᵤ = A.*ĉⁿᵤ .+ B.*fft(2.0*cᵤ³.*(cᵤ³.*cᵤ .- 1.0) - 0.25*λ_.*(cₗ .- cᵤ) .+ H₁) + C.*f̂ᵤ
        #ĉⁿ⁺¹ₗ = A.*ĉⁿₗ .+ B.*fft(2.0*cₗ³.*(cₗ³.*cₗ .- 1.0) - 0.25*λ_.*(cᵤ .- cₗ) .+ H₂) + C.*f̂ₗ
        # inverse Fourier transform
        cᵤ = real(ifft(ĉⁿ⁺¹ᵤ))
        cₗ = real(ifft(ĉⁿ⁺¹ₗ))
        # save
        if (save_every != 0) && (i % save_every == 0)
            push!(coupling_list_save, coupling_ratio(cᵤ, cₗ))
            push!(fit_list_save, domain_fit(cᵤ, cₗ))
            push!(c_list, [cᵤ, cₗ])
            # set progressbar title to coupling ratio
            if verbose
                pbar.desc = @sprintf("%.3f %.3f %.3f %.3f", coupling_list_plot[end], λ_, maximum(cᵤ), minimum(cᵤ))
            end
        end
        if (plot_every != 0) && (i % plot_every == 0)
            push!(coupling_list_plot, coupling_ratio(cᵤ, cₗ))
            push!(fit_list_plot, domain_fit(cᵤ, cₗ))
            p = plot_spectrum(cᵤ, cₗ, coupling_list_plot, fit_list_plot, step = plot_every)
            savefig("curr.png")
            if verbose
                pbar.desc = @sprintf("%.3f %.3f %.3f %.3f", coupling_list_save[end], λ_, maximum(cᵤ), minimum(cᵤ))
            end
        end
        if verbose
            next!(pbar)
        end
        λ_ = max(λ_ + dλdt * Δt, 0.0)
    end
    return c_list, coupling_list_save, fit_list_save
end

function animate_time_evolution(c_list, coupling_list, fit_list, filename="CHSolver.gif", ; step = 1.0)
    anim = @animate for i in eachindex(c_list)
        p = plot_spectrum(c_list[i][1], c_list[i][2], coupling_list[1:i], fit_list[1:i], step = step)
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
    ∂₁v = real(ifft(im*k.*v̂))
    return ∂₁v
end

function energy(cᵤ, cₗ)
    # compute energy
    #cᵤ⁶ = cᵤ.^6 
    #cₗ⁶ = cₗ.^6
    #cᵤ⁴ = cᵤ.^4
    #cₗ⁴ = cₗ.^4
    diff_cᵤ = diff(cᵤ)
    diff_cₗ = diff(cₗ)
    println()
    #enᵤ = (ϵ*0.25*sum((cᵤ⁶ .- 1.0).^2) + 0.5*K*sum(diff_c₁ᵤ.^2 + diff_c₂ᵤ.^2) + 0.125*λ*sum((cᵤ .- cₗ).^2) + 0.5 * sum((cᵤ.+1.0).*H₁ˡᵒ - (cᵤ .- 1.0).*H₁ˡᵈ))*(L/N)^2 
    #enₗ = (ϵ*0.25*sum((cₗ⁶ .- 1.0).^2) + 0.5*K*sum(diff_c₁ₗ.^2 + diff_c₂ₗ.^2) + 0.125*λ*sum((cₗ .- cᵤ).^2) + 0.5 * sum((cₗ.+1.0).*H₂ˡᵒ - (cₗ .- 1.0).*H₂ˡᵈ))*(L/N)^2
    #enᵤ = (ϵ*0.25*sum((cᵤ⁴ .- 1.0).^2) + 0.5*K*sum(diff_c₁ᵤ.^2 + diff_c₂ᵤ.^2) + 0.125*λ*sum((cᵤ .- cₗ).^2) + 0.5 * sum((cᵤ.+1.0).*H₁ˡᵒ - (cᵤ .- 1.0).*H₁ˡᵈ))*(L/N)^2
    #enₗ = (ϵ*0.25*sum((cₗ⁴ .- 1.0).^2) + 0.5*K*sum(diff_c₁ₗ.^2 + diff_c₂ₗ.^2) + 0.125*λ*sum((cₗ .- cᵤ).^2) + 0.5 * sum((cₗ.+1.0).*H₂ˡᵒ - (cₗ .- 1.0).*H₂ˡᵈ))*(L/N)^2
    enᵤ = (ϵ*0.25*sum((cᵤ.^2 .- 1.0).^2) + 0.5*K*sum(diff_cᵤ.^2) + 0.125*λ*sum((cᵤ .- cₗ).^2) + 0.5 * sum((cᵤ.+1.0).*H₁ˡᵒ - (cᵤ .- 1.0).*H₁ˡᵈ))*(L^2/N)
    enₗ = (ϵ*0.25*sum((cₗ.^2 .- 1.0).^2) + 0.5*K*sum(diff_cₗ.^2) + 0.125*λ*sum((cₗ .- cᵤ).^2) + 0.5 * sum((cₗ.+1.0).*H₂ˡᵒ - (cₗ .- 1.0).*H₂ˡᵈ))*(L^2/N)
    return enᵤ + enₗ
end


function make_grid(c₀,; upper=true, to_mid=false)
    # make initial zero grid
    c = zeros(N)
    if to_mid
        i = round(Int, 0.5*(1.0+c₀)*N)
        c[1:i] .= 1.0
        # set the rest to -1.0
        c[i+1:end] .= -1.0
        # shift by N/4
        c = circshift(c, (N÷4))
        c .+= c₀ - mean(c)
        return c
    end
    if upper
        # set 0.5+(1.0+c₀) of upper as 1.0
        i = round(Int, 0.5*(1.0+c₀)*N)
        c[1:i] .= 1.0
        # set the rest to -1.0
        c[i+1:end] .= -1.0
    else
        # set 0.5+(1.0-c₀) of upper as -1.0
        i = round(Int, 0.5*(1.0-c₀)*N)
        c[1:i] .= -1.0
        # set the rest to 1.0
        c[i+1:end] .= 1.0
    end
    c .+= c₀ - mean(c)
    return c
end

# find suggested lambda
function en_profile_1d()
    en_list = Vector{Float64}[]
    cᵤ = make_grid(c₀, upper=true)
    cₗ = make_grid(c₀, upper=true)
    @showprogress for i in 1:N
        push!(en_list, Float64[])
        # shift upper by 1 in the y direction
        cᵤ = circshift(cᵤ, 1)
        for i in 1:N
            # shift lower by 1 in the y direction
            cₗ = circshift(cₗ, 1)
            # minimize energy a bit
            c_list, coupling_list = solver(cᵤ, cₗ, 10, save_every=10, plot_every=0, initial_λ=0.0, noisy = false, verbose = false)
            push!(en_list[end], energy(c_list[end][1], c_list[end][2]))
        end
    end
    # plot energy heatmap
    en_list = hcat(en_list...)
    en_list .-= minimum(en_list)
    p = heatmap(en_list, aspect_ratio=1, color=:bluesreds, cbar=true, axis=nothing, border=:none, ticks=nothing, title = "Energy")
    # mark the minimas regions up to 3kT 
    for i in 1:N
        for j in 1:N
            if en_list[i,j] < 1.0
                # make with a red dot
                plot!(p, [j], [i], m=(5, :red), legend=false)
            end
        end
    end
    savefig("en_profile_1d.png")
end

en_profile_1d()

# make initial uniform grid
cᵤ = make_grid(c₀, to_mid = true)
cₗ = make_grid(c₀, to_mid = true)
#cᵤ, cₗ = load_state("final_large2.jld2")

# solve
#c_list, coupling_list = solver(cᵤ, cₗ, 1000000, save_every=0, plot_every=25)
#c_list, coupling_list = solver(cᵤ, cₗ, round(Int, 250/Δt), save_every=round(Int, 0.5/Δt), plot_every=round(Int, 0.5/Δt))
c_list, coupling_list, fit_list = solver(cᵤ, cₗ, round(Int, 100000/Δt), save_every=round(Int, 10/Δt), plot_every=round(Int, 100/Δt))
#c_list, coupling_list = solver(cᵤ, cₗ, 100000, save_every=10000, plot_every=0)
save_state(c_list[end][1], c_list[end][2], "1D.jld2")
animate_time_evolution(c_list, coupling_list, fit_list, "1D.gif", step = 10.0)