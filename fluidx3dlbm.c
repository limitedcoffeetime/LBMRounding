//This is the LBM core of the FLuidX3D OpenCL C implementation (D3Q19 SRT FP32/xx)
//This should only be used as reference to see how to implement the LBM core in Julia
//Remember: The following code does not include stochastic rounding.
float __attribute__((always_inline)) sq(const float x) {
    return x * x;
}

uint3 __attribute__((always_inline)) coordinates(const uint n) { // disassemble 1D index to 3D coordinates (n -> x,y,z)
    const uint t = n % (def_sx * def_sy);
    return (uint3)(t % def_sx, t / def_sx, n / (def_sx * def_sy)); // n = x + (y + z * sy) * sx
}

uint __attribute__((always_inline)) f_index(const uint n, const uint i) { // 32-bit indexing (maximum box size for D3Q19: 608x608x608)
    return i * def_s + n; // SoA (229% faster on GPU compared to AoS)
}

void __attribute__((always_inline)) equilibrium(const float rho, float ux, float uy, float uz, float *feq) { // calculate f_equilibrium
    const float c3 = -3.0f * (sq(ux) + sq(uy) + sq(uz)), rhom1 = rho - 1.0f; // c3 = -2*sq(u)/(2*sq(c))
    ux *= 3.0f;
    uy *= 3.0f;
    uz *= 3.0f;
    feq[0] = def_w0 * fma(rho, 0.5f * c3, rhom1); // 000 (identical for all velocity sets)
    const float u0 = ux + uy, u1 = ux + uz, u2 = uy + uz, u3 = ux - uy, u4 = ux - uz, u5 = uy - uz;
    const float rhos = def_ws * rho, rhoe = def_we * rho, rhom1s = def_ws * rhom1, rhom1e = def_we * rhom1;
    feq[1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
    feq[3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
    feq[5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
    feq[7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
    feq[9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
    feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), u2), rhom1e); feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
    feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), u3), rhom1e); feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
    feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), u4), rhom1e); feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
    feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), u5), rhom1e); feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
}

void __attribute__((always_inline)) fields(const float *f, float *rhon, float *uxn, float *uyn, float *uzn) { // calculate density and velocity from fi
    float rho = f[0], ux, uy, uz;
    #pragma unroll
    for (uint i = 1; i < def_set; i++) rho += f[i]; // calculate density from f
    rho += 1.0f; // add 1.0f last to avoid digit extinction effects when summing up f
    ux = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[15] - f[16]; // calculate velocity from fi (alternating + and - for best accuracy)
    uy = f[3] - f[4] + f[7] - f[8] + f[11] - f[12] + f[14] - f[13] + f[17] - f[18];
    uz = f[5] - f[6] + f[9] - f[10] + f[11] - f[12] + f[16] - f[15] + f[18] - f[17];
    *rhon = rho;
    *uxn = ux / rho;
    *uyn = uy / rho;
    *uzn = uz / rho;
}

void __attribute__((always_inline)) neighbors(const uint n, uint *j) { // calculate neighbor indices
    const uint3 xyz = coordinates(n);
    const uint x0 = xyz.x; // pre-calculate indices (periodic boundary conditions on simulation box walls)
    const uint xp = (xyz.x + 1) % def_sx;
    const uint xm = (xyz.x + def_sx - 1) % def_sx;
    const uint y0 = xyz.y * def_sx;
    const uint yp = ((xyz.y + 1) % def_sy) * def_sx;
    const uint ym = ((xyz.y + def_sy - 1) % def_sy) * def_sx;
    const uint z0 = xyz.z * def_sy * def_sx;
    const uint zp = ((xyz.z + 1) % def_sz) * def_sy * def_sx;
    const uint zm = ((xyz.z + def_sz - 1) % def_sz) * def_sy * def_sx;
    j[0] = n;
    j[1] = xp + y0 + z0; j[2] = xm + y0 + z0; // +00 -00
    j[3] = x0 + yp + z0; j[4] = x0 + ym + z0; // 0+0 0-0
    j[5] = x0 + y0 + zp; j[6] = x0 + y0 + zm; // 00+ 00-
    j[7] = xp + yp + z0; j[8] = xm + ym + z0; // ++0 --0
    j[9] = xp + y0 + zp; j[10] = xm + y0 + zm; // +0+ -0-
    j[11] = x0 + yp + zp; j[12] = x0 + ym + zm; // 0++ 0--
    j[13] = xp + ym + z0; j[14] = xm + yp + z0; // +-0 -+0
    j[15] = xp + y0 + zm; j[16] = xm + y0 + zp; // +0- -0+
    j[17] = x0 + yp + zm; j[18] = x0 + ym + zp; // 0+- 0-+
}

kernel void initialize(global fpXX *fc, global float *rho, global float *u) {
    const uint n = get_global_id(0); // n = x + (y + z * sy) * sx
    float feq[def_set]; // f_equilibrium
    equilibrium(rho[n], u[n], u[def_s + n], u[2 * def_s + n], feq);
    #pragma unroll
    for (uint i = 0; i < def_set; i++) store(fc, f_index(n, i), feq[i]); // write to fc
} // initialize()

kernel void stream_collide(const global fpXX *fc, global fpXX *fs, global float *rho, global float *u, global uchar *flags) {
    const uint n = get_global_id(0); // n = x + (y + z * sy) * sx
    const uchar flagsn = flags[n]; // cache flags[n] for multiple readings
    if (flagsn & TYPE_W) return; // if node is boundary node, just return (slight speed up)
    uint j[def_set]; // neighbor indices
    neighbors(n, j); // calculate neighbor indices
    uchar flagsj[def_set]; // cache neighbor flags for multiple readings
    flagsj[0] = flagsn;
    #pragma unroll
    for (uint i = 1; i < def_set; i++) flagsj[i] = flags[j[i]];
    // read from fc in video memory and stream to fhn
    float fhn[def_set]; // cache f_half_step[n], do streaming step
    fhn[0] = fc[f_index(n, 0)]; // keep old center population
    #pragma unroll
    for (uint i = 1; i < def_set; i += 2) { // perform streaming
        fhn[i] = load(fc, flagsj[i + 1] & TYPE_W ? f_index(n, i + 1) : f_index(j[i + 1], i)); // boundary : regular
        fhn[i + 1] = load(fc, flagsj[i] & TYPE_W ? f_index(n, i) : f_index(j[i], i + 1));
    }
    // collide fh
    float rhon, uxn, uyn, uzn; // cache density and velocity for multiple writings/readings
    fields(fhn, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from f
    uxn = clamp(uxn, -def_c, def_c); // limit velocity (for stability purposes)
    uyn = clamp(uyn, -def_c, def_c);
    uzn = clamp(uzn, -def_c, def_c);
    float feq[def_set]; // cache f_equilibrium[n]
    equilibrium(rhon, uxn, uyn, uzn, feq); // calculate equilibrium populations
    #pragma unroll
    for (uint i = 0; i < def_set; i++) store(fs, f_index(n, i), fma(1.0f - def_w, fhn[i], def_w * feq[i])); // write to fs in video memory
} // stream_collide()
