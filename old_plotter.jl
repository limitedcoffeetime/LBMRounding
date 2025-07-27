using JLD2
using Plots

# 1) Load data from JLD2
file = jldopen("all_data_final.jld2", "r")
all_data = file["all_data"]
close(file)

# 2) Theoretical curve and constants
function get_constants(::Type{T}) where T
    scale = 8
    NX = 32 * scale
    NY = NX
    ndir = 9
    w0 = T(4.0) / T(9.0)
    ws = T(1.0) / T(9.0)
    wd = T(1.0) / T(36.0)
    wi = T[w0, ws, ws, ws, ws, wd, wd, wd, wd]
    dirx = [0,  1,  0, -1,  0,  1, -1, -1,  1]
    diry = [0,  0,  1,  0, -1,  1,  1, -1, -1]
    nu = T(1.0) / T(6.0)
    tau = T(3.0) * nu + T(0.5)
    omega = T(1.0) / tau
    u_max = T(0.04) / T(scale)
    rho0 = T(1.0)
    NSTEPS = 200 * scale * scale
    return (
        scale=scale, NX=NX, NY=NY, ndir=ndir,
        w0=w0, ws=ws, wd=wd, wi=wi,
        dirx=dirx, diry=diry,
        nu=nu, tau=tau, omega=omega,
        u_max=u_max, rho0=rho0, NSTEPS=NSTEPS
    )
end

function theoretical_energy_decay(n_steps::Int, c)
    kx = 2π / c.NX
    ky = 2π / c.NY
    td = 1.0 / (c.nu * (kx^2 + ky^2))
    return [exp(-2.0 * t / td) for t in 1:n_steps]
end

const nsteps = 200_000
const c64    = get_constants(Float64)
const theory = theoretical_energy_decay(nsteps, c64)

# 3) Parse each key into (bitwidth, rounding, method)
function parse_key(key::AbstractString)
    pattern = r"^Float(16|32|64)(sr)?_(std|optimized)$"
    m = match(pattern, key)
    if m === nothing
        error("Key '$key' does not match the pattern Float(16|32|64)(sr)?_(std|optimized).")
    end
    bits_str   = m[1]            # "16"/"32"/"64"
    sr_capture = m[2]            # "sr" or nothing
    method_str = m[3]            # "std" or "optimized"

    # sr => "stochastic", else "deterministic"
    rounding_str = (sr_capture == "sr") ? "stochastic" : "deterministic"
    return (bits_str, rounding_str, method_str)
end

# 4) Define color and style mappings
color_map = Dict(
    ("16", "deterministic") => :blue,
    ("16", "stochastic")    => :cyan,
    ("32", "deterministic") => :red,
    ("32", "stochastic")    => :orange,
    ("64", "deterministic") => :green,
    ("64", "stochastic") => :lightgreen,# no sr for 64
)

function line_style(method::AbstractString)
    return (method == "std") ? :dash : :solid
end

function make_label(bits::AbstractString, rounding::AbstractString, method::AbstractString)
    return "FP$(bits) ($(rounding), $(method))"
end

# 5) Build and display the plot
default(legend=:bottomleft, linewidth=2, framestyle=:box)

p = plot(
    yscale = :log10,
    ylims  = (1e-40, 1e2),
    xlims  = (1, nsteps),
    xlabel = "Timestep",
    ylabel = "E(t)/E₀",
    title  = "Taylor Green Vortex Decay Comparison",
    yticks = (10.0 .^ range(-40, stop=2, step=5)),
    minorgrid = false,
    right_margin = Plots.mm * 10,
    dpi = 200
)

# Downsample so dashed lines remain visually distinct
plot_step = 200
idx       = 1:plot_step:nsteps

for key in sort(collect(keys(all_data)))
    data = all_data[key]
    bits, rounding, method = parse_key(key)
    c   = color_map[(bits, rounding)]
    lsy = line_style(method)
    lbl = make_label(bits, rounding, method)

    plot!(p, idx, data[idx];
          color=c, linestyle=lsy, label=lbl)
end

# Add theoretical line (dashed black)
plot!(p, idx, theory[idx];
      color=:black, linestyle=:dash, label="Theoretical")

display(p)
