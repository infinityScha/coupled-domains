using DataFrames
using CSV
using ProgressMeter
using Printf
using JLD2
using Statistics

const M₁ = 7 # log(2, grid size)
const M₂ = 6 # log(2, grid size)
# deduced numerical & physical parameters
const N₁ = 2^M₁ # grid size
const N₂ = 2^M₂ # grid size
const _R = 1.0
const Jₛᴸᵒ = -0.30 # spontaneous curvature of Lo phase [nm^-1]
const Jₛᴸᵈ = -0.30 # spontaneous curvature of Ld phase [nm^-1]
const Kᴸᵒ = 23.0 # bending modulus of Lo phase [kT]
const Kᴸᵈ = 9.0 # bending modulus of Ld phase [kT]  

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
        H₁ˡᵒ[i,j] = Hij(i, _R)
        H₁ˡᵈ[i,j] = Hij(i, _R, false)
        H₂ˡᵒ[i,j] = Hij(i, -_R)
        H₂ˡᵈ[i,j] = Hij(i, -_R, false)
    end
end

const H₁ = 0.5*(H₁ˡᵒ .- H₁ˡᵈ)
const H₂ = 0.5*(H₂ˡᵒ .- H₂ˡᵈ)


function _coupling_ratio(cᵤ, cₗ)
    agreement = (cᵤ .* cₗ) .> 0.0
    return mean(agreement)
    #return 1.0 - 0.5*mean(abs.(cᵤ .- cₗ))
end

function _domain_fit(cᵤ, cₗ)
    fitᵤ = cᵤ.*(H₁.-mean(H₁)) .< 0.0
    fitₗ = cₗ.*(H₂.-mean(H₂)) .< 0.0
    return mean(fitᵤ), mean(fitₗ)
end


function make_dataframe(filename)
    # Define your parameter lists
    which_lipid = ["DAPC", "DPPC"]
    gammas = [0.5, 1.0, 2.0, 4.0]
    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0]
    Rs = [100.0, 75.0, 50.0, 40.0, 30.0, 27.5, 25.0, 22.5, 20.0, 17.5, 15.0, 12.5, 10.0, 7.5, 5.0]
    c0s = [0.0]
    is_mids = ["true", "false"]
    runs = 1:50

    # Create a DataFrame with the specified columns
    columns = [:which_lipid, :gamma, :lambda, :R, :c0, :is_mid, :run, :coupling_ratio, :domain_fit_ᵤ, :domain_fit_ₗ]
    df = DataFrame([Vector{Any}(undef, 0) for _ in columns], columns)

    # Calculate the total number of iterations
    total_iterations = length(which_lipid) * length(gammas) * length(lambdas) * length(Rs) * length(c0s) * length(is_mids) * length(runs)

    # Initialize the progress bar
    p = Progress(total_iterations)

    for (lipid, gamma, lambda, R, c0, is_mid, run) in Iterators.product(which_lipid, gammas, lambdas, Rs, c0s, is_mids, runs)
        # Construct the path to the .jld2 file
        path = joinpath("new_model", lipid, string(gamma), string(lambda), string(R), string(c0), is_mid, string(run) * ".jld2")

        # Load the data, keep only last 10 frames
        c_list = load(path, "c_list")[end-19:end]

        # Calculate the coupling ratio
        for c in c_list
            coupling_ratio = _coupling_ratio(c[1], c[2])
            domain_fit1, domain_fit2 = _domain_fit(c[1], c[2])
            df = push!(df, [lipid, gamma, lambda, R, c0, is_mid, run, coupling_ratio, domain_fit1, domain_fit2])
        end
        next!(p)
    end
    # save the DataFrame to a .csv file
    CSV.write(filename, df) 
end

make_dataframe("new_model/analysis_df.csv")