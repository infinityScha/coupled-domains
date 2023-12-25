using Statistics
using Plots
using ProgressMeter

# physical parameters
const γ = 1.0 # line tension [kT/nm] = K*4/(3*sqrt(2K/ϵ)) = 8*K/3*σ -> K = 3*γ/8/σ 
const σ = 1.5 # interface width [nm]
const m = 1.0 # mobility [nm^2/ns/kT]
const c₀ = 0.0 # bulk field value
# numerical parameters
const Nx, Ny = 100, 200 # grid size
const dh = 0.1 # grid spacing [nm]
const dt = 0.001 # time step [ns]
# deduced physical parameters
temp = 3*γ/8/σ
const ϵ = 1.0#8*temp/σ*σ
const K = 1.0#temp/dh/dh # scaled line tension
println("K = ", K, " [kT/nm^2]", ", ϵ = ", ϵ, " [kT/nm^2]")

function laplacian(c)
    # compute laplacian of c
    lap_c = zeros(Nx,Ny)
    lap_c[2:end-1,2:end-1] = c[1:end-2,2:end-1]+c[3:end,2:end-1]+c[2:end-1,1:end-2]+c[2:end-1,3:end]-4*c[2:end-1,2:end-1]
    # also do on boundaries
    lap_c[1, 2:end-1] = c[2, 2:end-1]+c[end, 2:end-1]+c[1, 1:end-2]+c[1, 3:end]-4*c[1, 2:end-1]
    lap_c[end, 2:end-1] = c[1, 2:end-1]+c[end-1, 2:end-1]+c[end, 1:end-2]+c[end, 3:end]-4*c[end, 2:end-1]
    lap_c[2:end-1, 1] = c[1:end-2, 1]+c[3:end, 1]+c[2:end-1, end]+c[2:end-1, 2]-4*c[2:end-1, 1]
    lap_c[2:end-1, end] = c[1:end-2, end]+c[3:end, end]+c[2:end-1, end-1]+c[2:end-1, 1]-4*c[2:end-1, end]
    # also do on corners
    lap_c[1, 1] = c[2, 1]+c[end, 1]+c[1, end]+c[1, 2]-4*c[1, 1]
    lap_c[end, 1] = c[1, 1]+c[end-1, 1]+c[end, end]+c[end, 2]-4*c[end, 1]
    lap_c[1, end] = c[2, end]+c[end, end]+c[1, end-1]+c[1, 1]-4*c[1, end]
    lap_c[end, end] = c[1, end]+c[end-1, end]+c[end, end-1]+c[end, 1]-4*c[end, end]
    return lap_c/dh/dh
end

function f(c)
    return m*laplacian(-ϵ*c+ϵ*c.*c.*c-K*laplacian(c))
end

function solver(c, Nsteps)
    c_list = [c]
    @showprogress for i in 1:Nsteps
        c = copy(c)
        # midpoint method
        c_mid = c .+ 0.5*dt*f(c)
        c .+= dt*f(c_mid)
        # save concentration
        push!(c_list, c)
    end
    return c_list
end

function animate_time_evolution(c_list)
    ratio = Ny/Nx
    anim = @animate for i in 1:length(c_list)
        heatmap(c_list[i], aspect_ratio=1, color=:viridis, cbar=false, clim=(-1,1))
        # remove axes and ticks
        plot!(size=(Int64(500*ratio),500), ticks=nothing, border=:none, axis=nothing)
    end
    gif(anim, "CHSolver.gif", fps=10)
end


# make initial uniform grid + random uniform noise
c = 1.0e-5*rand(Nx,Ny)
c .+= c₀ - mean(c)
# solve
c_list = solver(c, 100)
#println(c_list[end])
animate_time_evolution(c_list)