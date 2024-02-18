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
const I = 0.85714*sqrt(2) # for f(c)=0.25*(c⁶-1)², for regular CH it is 2/3*sqrt(2)
const γ = 3.0 # line tension [kT/nm] = I√(K⋅ϵ)
const σ = 1.0 # interface width [nm] = √(K/ϵ)
const ζ = 4.9e-2 # Cook noise [kT/ns]
const mc_step_size = 0.01 # monte carlo step size
# numerical parameters
const M = 8 # log(2, grid size)
const L = 20.0 # box size [nm]
const Δt = 1.0e-4 # time step [ns]
# deduced numerical & physical parameters
const N = 2^M # grid size
const K = γ*σ/I
const ϵ = K/σ^2


# deduced Fourier parameters
k = (2.0*π/L)*cat(0:N/2-1, -N/2:-1, dims=1)
k² = k.^2
k⁴ = k².^2

# deduced semi-implicit parameters
temp = 1.0 .+ K*Δt*k⁴
const A = 1.0./temp
const B = -ϵ*Δt*k²./temp
const C = -Δt./temp
const η = sqrt(2.0*ζ/Δt/L*N) # NOTE! 1D!

println("K = ", K, " [kT]", ", ϵ = ", ϵ, " [kT/nm^2]")

function plot_spectrum(c, label, ; p = nothing)
    # Create the profile
    if p == nothing
        p = plot(LinRange(0.0, L, N), c, xlabel="x [nm]", ylabel="c", label=label, linewidth=2)
    else
        plot!(p, LinRange(0.0, L, N), c, label=label)
    end
    return p
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

function make_grid()
    # make initial zero grid
    c = zeros(N)
    # set left as -1.0
    i = round(Int, 0.5*N)
    c[1:i] .= -1.0
    # set the rest to 1.0
    c[i+1:end] .= 1.0
    # shift by N/4
    c = circshift(c, round(Int, 0.25*N))
    return c
end

function solver_noise(c, Nsteps, ;save_every=100, plot_every=0, noisy=true)
    c_list = [c]
    pbar = Progress(Nsteps)
    for i in 1:Nsteps
        # Fourier transform
        ĉⁿ = fft(c)
        # random noise
        f̂ = noisy ? noise() : zeros(N)
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
            p = plot_spectrum(c, "Noise")
            savefig("curr.png")
        end
        next!(pbar)
    end
    return c_list
end

function energy(c)
    # compute energy over LxL box
    c⁶ = c.^6
    return (ϵ*0.25*sum((c⁶ .- 1.0).^2) + 0.5*K*sum((diff(c)).^2))/N*L*L
end

function solver_mc(c, Nsteps, ;save_every=100, plot_every=0)
    c_list = [c]
    curr_c = copy(c)
    curr_en = energy(c)
    prev_c = copy(c)
    prev_en = curr_en
    accepted = 0
    total_steps = 0
    accepted_vec = zeros(Int(N/2))
    total_vec = zeros(Int(N/2))
    total_vec[1] = 1
    pbar = Progress(Nsteps)
    for i in 1:Nsteps
        for j in 1:N
            prev_c = copy(curr_c)
            prev_en = curr_en
            # need to change both real and imaginary part
            curr_c[j] += mc_step_size*(2*rand()-1)
            # compute energy difference
            curr_en = energy(curr_c)
            # accept or reject
            if curr_en > prev_en
                # kT = 1
                if rand() > exp(prev_en - curr_en)
                    # reject
                    curr_c = copy(prev_c)
                    curr_en = prev_en
                else
                    accepted += 1
                end
            else
                accepted += 1
            end
            total_steps += 1
        end
        next!(pbar)
        # save
        if (save_every != 0) && (i % save_every == 0)
            push!(c_list, curr_c)
            # update progress bar description to the acceptance rate and energy
            pbar.desc = @sprintf("Acceptance rate: %.3f, Energy: %.3f", accepted/total_steps, curr_en)
        end
        if (plot_every != 0) && (i % plot_every == 0)
            p = plot_spectrum(curr_c, "MC")
            savefig("curr.png")
            # plot acceptance rate
            p = plot(accepted_vec./total_vec, xlabel="x [nm]", ylabel="Acceptance rate", label="Acceptance rate", linewidth=2, p=p)
            savefig("acceptance_rate.png")
        end
    end
    return c_list
end

function diff(v)
    # to in fourier space
    v̂ = fft(v)
    # multiply by k
    v̂[1] = 0.0
    v̂[2:end] .= v̂[2:end].*k[2:end].*im
    # back to real space
    return real(ifft(v̂))
end

function noise()
    f = randn(N)
    # compute divergence of random vector field
    f̂ = fft(diff(f)*η)
    # keep only fluctuations with k > 1 nm -1
    #f̂[k² .> (4.0*π*π)] .= 0.0
    return f̂
end

function interface_ave_profile(c_list)
    # compute average interface profile
    c̄ = zeros(N)
    for c in c_list
        c̄ .+= c
    end
    c̄ ./= length(c_list)
    return c̄
end

# run non-noise relaxation
c = make_grid()
c_list = solver_noise(c, 10000, save_every=100, plot_every=1000, noisy=false)
c = c_list[end]
println("Energy: ", energy(c), "expected: ", 2*γ*L)
# run mc simulation
c_list_mc = solver_mc(c, 25000, save_every=10, plot_every=1000)
p = plot_energy_kernel_density(c_list_mc[100:end], "MC")
# run noise simulation
c = c_list[1]
c_list_noise = solver_noise(c, 100000, save_every=10, plot_every=1000)
p = plot_energy_kernel_density(c_list_noise[100:end], "Noise", p=p)
savefig("interface_ave_profile.png")