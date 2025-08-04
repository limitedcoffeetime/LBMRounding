using StochasticRounding
using Plots
using ProgressMeter
using Random

# 1. SIMULATION FUNCTIONS & HELPERS

# -- Get simulation constants --
function get_constants(::Type{T}) where T
    scale = 8                   # Resolution scale factor
    NX = 32 * scale             # Domain size in x
    NY = NX                     # Domain size in y
    ndir = 9                    # Number of directions
    w0 = T(4.0) / T(9.0)        # Center weight
    ws = T(1.0) / T(9.0)        # Cardinal weights
    wd = T(1.0) / T(36.0)       # Diagonal weights
    wi = T[w0, ws, ws, ws, ws, wd, wd, wd, wd]
    dirx = [0,  1,  0, -1,  0,  1, -1, -1,  1]
    diry = [0,  0,  1,  0, -1,  1,  1, -1, -1]
    nu = T(1.0) / T(6.0)
    tau = T(3.0) * nu + T(0.5)
    omega = T(1.0) / tau
    u_max = T(0.04) / T(scale)
    rho0 = T(1.0)
    NSTEPS = 200 * scale * scale  # Default total timesteps
    return (; scale, NX, NY, ndir, w0, ws, wd, wi, dirx, diry,
            nu, tau, omega, u_max, rho0, NSTEPS)
end

# -- Indexing functions for 1D storage --
@inline function scalar_index(x::Int, y::Int, constants)::Int
    return y * constants.NX + x + 1
end

@inline function field_index(x::Int, y::Int, d::Int, constants)::Int
    return (d * constants.NY + y) * constants.NX + x + 1
end

# -- Taylor–Green vortex initialization --
function taylor_green_point(t::Int, x::Int, y::Int, constants)
    kx = 2π / constants.NX
    ky = 2π / constants.NY
    td = 1.0 / (constants.nu * (kx*kx + ky*ky))
    X = x + 0.5
    Y = y + 0.5
    ux = -constants.u_max * sqrt(ky / kx) * cos(kx * X) * sin(ky * Y) * exp(-t / td)
    uy =  constants.u_max * sqrt(kx / ky) * sin(kx * X) * cos(ky * Y) * exp(-t / td)
    P = -0.25 * constants.rho0 * (constants.u_max*constants.u_max) *
        ((ky / kx) * cos(2.0 * kx * X) + (kx / ky) * cos(2.0 * ky * Y)) *
        exp(-2.0 * t / td)
    rho = constants.rho0 + 3.0 * P
    return rho, ux, uy
end

function taylor_green(t::Int, r::Vector{T}, u::Vector{T}, v::Vector{T}, constants) where T
    @inbounds for y in 0:constants.NY-1
        @simd for x in 0:constants.NX-1
            let idx = scalar_index(x, y, constants)
                rho, ux, uy = taylor_green_point(t, x, y, constants)
                r[idx] = T(rho)
                u[idx] = T(ux)
                v[idx] = T(uy)
            end
        end
    end
end

# -- Compute kinetic energy with Float64 accumulator --
function kinetic_energy(u::AbstractVector{T}, v::AbstractVector{T}) where T
    s = 0.0                        # 64-bit accumulator
    @inbounds @simd for i in eachindex(u)
        ui = float(u[i]); vi = float(v[i])
        s += ui*ui + vi*vi
    end
    return s * 0.5
end



@inline function equilibrium_shifted(rho::T, ux::T, uy::T, w::T, cx::T, cy::T) where T
    cidotu = cx*ux + cy*uy           # c_i · u
    usqr   = ux*ux + uy*uy           # u · u

    velocity_poly = T(3.0)*cidotu +
    T(4.5)*cidotu*cidotu -
    T(1.5)*usqr

    return w * rho * velocity_poly +          # ρ‑weighted velocity part
    w * (rho - one(T))                 # mass‑shift term
end


function init_equilibrium!(f::Vector{T}, r::Vector{T}, u::Vector{T}, v::Vector{T}, constants;
                           use_ddf_shift::Bool=false) where T
    @inbounds for y in 0:constants.NY-1
        @simd for x in 0:constants.NX-1
            let sidx = scalar_index(x, y, constants)
                rho = r[sidx]
                uxv = u[sidx]
                uyv = v[sidx]
                for i in 0:constants.ndir-1
                    let fidx = field_index(x, y, i, constants)
                        w = constants.wi[i+1]
                        cx = T(constants.dirx[i+1])
                        cy = T(constants.diry[i+1])
                        if use_ddf_shift
                            f[fidx] = equilibrium_shifted(rho, uxv, uyv, w, cx, cy)
                        else
                            let cidotu = cx * uxv + cy * uyv
                                usqr = uxv*uxv + uyv*uyv
                                f[fidx] = w * rho * (one(T) + T(3.0)*cidotu + T(4.5)*(cidotu*cidotu) - T(1.5)*usqr)
                            end
                        end
                    end
                end
            end
        end
    end
end

# -- Streaming step (with periodic boundaries) --
function stream!(f_src::Vector{T}, f_dst::Vector{T}, constants) where T
    @inbounds for d in 0:constants.ndir-1
        for y in 0:constants.NY-1
            @simd for x in 0:constants.NX-1
                let src_x = (x - constants.dirx[d+1] + constants.NX) % constants.NX
                    src_y = (y - constants.diry[d+1] + constants.NY) % constants.NY
                    f_dst[field_index(x, y, d, constants)] =
                        f_src[field_index(src_x, src_y, d, constants)]
                end
            end
        end
    end
end

# -- Compute macroscopic fields (density and velocity) --
function compute_rho_u!(f::Vector{T}, r::Vector{T}, u::Vector{T}, v::Vector{T}, constants;
                        use_ddf_shift::Bool=false, use_alt_sum::Bool=false) where T
    @inbounds for y in 0:constants.NY-1
        for x in 0:constants.NX-1
            let fvals = ntuple(i -> f[field_index(x, y, i-1, constants)], constants.ndir)
                if !use_ddf_shift
                    rho_local = zero(T)
                    ux_local  = zero(T)
                    uy_local  = zero(T)
                    if use_alt_sum
                        let (f0, f1, f2, f3, f4, f5, f6, f7, f8) = fvals
                            rho_local = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
                            ux_local = (f1 - f3) + (f5 - f7) + (f8 - f6)
                            uy_local = (f2 - f4) + (f5 + f6) - (f7 + f8)
                        end
                    else
                        for i in 0:constants.ndir-1
                            let fi = fvals[i+1]
                                rho_local += fi
                                ux_local += T(constants.dirx[i+1]) * fi
                                uy_local += T(constants.diry[i+1]) * fi
                            end
                        end
                    end
                    let sidx = scalar_index(x, y, constants)
                        r[sidx] = rho_local
                        u[sidx] = ux_local / rho_local
                        v[sidx] = uy_local / rho_local
                    end
                else
                    let (f0, f1, f2, f3, f4, f5, f6, f7, f8) = fvals
                        local rho_local, ux_local, uy_local
                        if use_alt_sum
                            rho_local = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
                            ux_local = (f1 - f3) + (f5 - f7) + (f8 - f6)
                            uy_local = (f2 - f4) + (f5 + f6) - (f7 + f8)
                        else
                            rho_local = zero(T)
                            ux_local = zero(T)
                            uy_local = zero(T)
                            for i in 0:constants.ndir-1
                                let fi = fvals[i+1]
                                    rho_local += fi
                                    ux_local += T(constants.dirx[i+1]) * fi
                                    uy_local += T(constants.diry[i+1]) * fi
                                end
                            end
                        end
                        let sidx = scalar_index(x, y, constants)
                            r[sidx] = rho_local + one(T)
                            u[sidx] = ux_local / r[sidx]
                            v[sidx] = uy_local / r[sidx]
                        end
                    end
                end
            end
        end
    end
end

# -- Collision step --
function collide!(f::Vector{T}, r::Vector{T}, u::Vector{T}, v::Vector{T}, constants;
                  use_ddf_shift::Bool=false) where T
    @inbounds for y in 0:constants.NY-1
        for x in 0:constants.NX-1
            let sidx = scalar_index(x, y, constants)
                let rho = r[sidx]
                    let ux = u[sidx]
                        let uy = v[sidx]
                            for i in 0:constants.ndir-1
                                let fidx = field_index(x, y, i, constants)
                                    let w = constants.wi[i+1]
                                        let cx = T(constants.dirx[i+1])
                                            let cy = T(constants.diry[i+1])
                                                if use_ddf_shift
                                                    let feq_shifted = equilibrium_shifted(rho, ux, uy, w, cx, cy)
                                                        f[fidx] = f[fidx] - constants.omega * (f[fidx] - feq_shifted)
                                                    end
                                                else
                                                    let cidotu = cx * ux + cy * uy
                                                        let usqr = ux*ux + uy*uy
                                                            let feq_unshifted = w * rho * (one(T) + T(3.0)*cidotu +
                                                                T(4.5)*(cidotu*cidotu) - T(1.5)*usqr)
                                                                f[fidx] = f[fidx] - constants.omega * (f[fidx] - feq_unshifted)
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# 2. SIMULATION & ENERGY RECORDING

"""
    run_simulation_energy(::Type{T}; steps, key, use_ddf_shift, use_alt_sum, seed)

Runs the Taylor–Green simulation for precision type `T` using the specified parameters.
It records the normalized kinetic energy E(t)/E₀ at every step.

Now the default configuration is to run with ddf_shift=true and alt_sum=true.
If seed is provided, sets the random seed for reproducible stochastic rounding.
"""
function run_simulation_energy(::Type{T}; steps::Int=-1, key::String="default",
                               use_ddf_shift::Bool=true, use_alt_sum::Bool=true,
                               seed::Integer = 42) where T
    rng = MersenneTwister(seed)
    Random.seed!(rng, seed)            # if plain rand() is ever used
    StochasticRounding.set_rng!(rng)   # drives Float*sr numbers

    constants = get_constants(T)
    if steps == -1
        steps = constants.NSTEPS
    end
    NX = constants.NX
    NY = constants.NY
    total_cells = NX * NY

    # Allocate arrays for the distribution functions and macroscopic fields
    f1 = zeros(T, total_cells * constants.ndir)
    f2 = zeros(T, total_cells * constants.ndir)
    rho = zeros(T, total_cells)
    ux  = zeros(T, total_cells)
    uy  = zeros(T, total_cells)

    # Initialize with Taylor–Green vortex conditions at t=0
    taylor_green(0, rho, ux, uy, constants)
    init_equilibrium!(f1, rho, ux, uy, constants; use_ddf_shift=use_ddf_shift)

    # Prepare recording array for energy evolution
    E0 = kinetic_energy(ux, uy)
    energies = Vector{Float64}(undef, steps)

    # Set up progress bar
    pb = Progress(steps, 1)
    for step in 1:steps
        stream!(f1, f2, constants)
        compute_rho_u!(f2, rho, ux, uy, constants;
                       use_ddf_shift=use_ddf_shift, use_alt_sum=use_alt_sum)
        collide!(f2, rho, ux, uy, constants; use_ddf_shift=use_ddf_shift)
        f1, f2 = f2, f1

        energies[step] = kinetic_energy(ux, uy) / E0
        next!(pb)
    end

    return energies, steps, constants, key
end

# 3. RUNNING SIMULATIONS & PLOTTING ENERGY EVOLUTION

# Define the precision types (Float32sr and Float16sr are provided by StochasticRounding)
precision_types = Dict(
     "Float64sr"   => Float64sr,
     "Float32"   => Float32,
     "Float16"   => Float16,
     "Float16sr" => Float16sr,
     "Float32sr" => Float32sr,


)

# Set simulation parameters (200,000 time steps)
steps = 200_000

# --- Choose simulation configuration ---
# Set these to true or false to choose your configuration.
# By default, we now run with alt_sum=true and ddf_shift=true.
use_ddf_shift = true
use_alt_sum   = true

# Dictionary to hold energy evolution data for each FP representation
energy_results = Dict{String, Vector{Float64}}()

# Run simulation for each FP representation using the chosen configuration
for (ptype_name, ptype) in precision_types
    # Build a key that reflects the configuration
    config_label = (use_ddf_shift ? "shift" : "unshift") * "_" * (use_alt_sum ? "alt" : "std")
    key = string(ptype_name, "_", config_label)
    println("Running simulation for $key ...")
    energies, sim_steps, constants, key = run_simulation_energy(ptype;
        steps=steps,
        key=key,
        use_ddf_shift=use_ddf_shift,
        use_alt_sum=use_alt_sum)
    energy_results[key] = energies
end

# Plot energy evolution for each FP representation
plt = plot(title="Normalized Energy Evolution E(t)/E₀",
           xlabel="Time step",
           ylabel="E(t)/E₀",
           legend=:bottomleft)
for (key, energies) in energy_results
    plot!(1:steps, energies, label=key)
end

display(plt)
