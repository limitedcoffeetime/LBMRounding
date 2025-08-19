# Exploring stochastic rounding for the lattice Boltzmann method

> **Note:** This blog post was originally authored by [Ryley McConkey](https://ryleymcconkey.com/),
> my research advisor. I’ve copied it here so the write-up is visible in this repository.
> The implementation described was my undergraduate research contribution.
>
> You can find the original blog post here: [Exploring stochastic rounding for the lattice Boltzmann method](https://ryleymcconkey.com/2025/07/stochastic_rounding_lbm/).

> *Original blog post follows below (copied with permission).*
---
*Note: this was an undergraduate research project that I supervised. The implementation for this project was completed by [David De Le Torre](https://www.linkedin.com/in/david-delatorre/), an undergraduate researcher at MIT. This post aims to both showcase David's awesome work, and provide utility to the LBM community.*

## Introduction

An important problem in modern computational fluid dynamics is whether or not we need to carry full double precision (FP64) accuracy. Most of the time, just end up carrying double precision. However, using single (FP32) or half (FP16) precision can exploit available architectures (e.g., [tensor cores](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)), reduce memory demands, and increase computational speed. It's an active area of research in CFD. See these recent articles which explore single and mixed precision for CFD:
- (2024) [On floating point precision in computational fluid dynamics using OpenFOAM, by Brogi et al.](https://www.sciencedirect.com/science/article/pii/S0167739X23003813)
- (2025) [Effects of lower floating-point precision on scale-resolving numerical simulations of turbulence, by Karp et al.](https://arxiv.org/pdf/2506.05150)


The effect of floating point representation on the accuracy of the lattice Boltzmann method (LBM) was studied fairly methodically by [Lehman et al. (2021)](https://arxiv.org/abs/2112.08926), the authors of the impressive [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) LBM code. [Here's FluidX3D being used to simulate turbulence around an X-Wing.](https://www.youtube.com/watch?v=3JNVBQyetMA). Essentially, Lehman et al. showed that error accumulation in FP16 representations is too high, and usually FP32 or FP64 is required. They also came up with their own version of FP32 design for the lattice Boltzmann method.

In this project, we aimed to conduct a similar analysis as Lehman et al. to investigate whether or not stochastic rounding can enable the use of lower accuracy floating point representations in LBM simulations. Check out this [blog post](https://nhigham.com/2020/07/07/what-is-stochastic-rounding/), [paper](https://hal.science/hal-03378080v1/document), and [Julia](https://github.com/milankl/StochasticRounding.jl) implementation of stochastic rounding for more details.


I think stochastic rounding is an interesting idea. Essentially, rather than always (i.e., deterministically) rounding $0.7\mapsto 1$, and $0.3 \mapsto 0$, we round them with some *probability* to these nearest values. For example, $1.3 \mapsto 1$ with probability 0.7, and $1.3 \mapsto 2$ with probability 0.3. The core idea of this rounding algorithm is that is results in a less biased roundoff error, since the rounding is random. For a computation with many iterations (e.g., a CFD simulation, or training a neural network), this results in an overall improvement in roundoff error.

The main research question in this investigation was: **for a lattice Boltzmann code, can stochastic rounding reduce round-off errors to the point of enabling the use of lower floating-point representations?**

Our main goals were to:
- Implementing a stochastic-rounding enabled 3D lattice Boltzmann method code in Julia
- Confirming the results of the above arXiv paper, without the use of stochastic rounding
- Determine if stochastic rounding enables tInvestigating the error differences when stochastic rounding is used with FP16

The main issue is whether error accumulation with repeated rounding can cause inaccurate simulation results.


## Implementation
David implemented a Julia-based LBM solver that allows the user to specify the floating point type used. David's code is available [here](https://github.com/limitedcoffeetime/LBMRounding/tree/main).

David implemented the vanilla LBM algorithm, as well as the DDF-shifted version used by Lehman et al. DDF shifting is essentially a clever arithmetic trick that greatly reduces accumulation of roundoff error in the LBM algorithm. You can read more about DDF shifting in [Lehman et al.'s paper](https://arxiv.org/abs/2112.08926).

## Results
We focused on simulating [Taylor-Green vortex decay](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex). This is a simple canonical flow used to test CFD solver implementations. It's one of the rare cases where we know the analytical solution to the Navier-Stokes equations given a set of initial and boundary conditions, so we can exactly evaluate the error in our simulation.

Here's essentially the "reference" simulation. It's a 64x64 grid simulation of Taylor Green vortex decay, with FP64 representation and DDF shifting used.
<img src="/images/taylor_green_vortex_Float64_shift.gif" alt="Description 2" style="width: 100%;">

Here are the results from experimenting with different floating point representations (FP32 and FP16), stochastic rounding (sr), and distribution shifting (shift/unshift). This is over the first few time steps of the decay.

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
    <img src="/images/taylor_green_vortex_Float32_unshift.gif" alt="Description 3" style="width: 100%;">
    <img src="/images/taylor_green_vortex_Float32_shift.gif" alt="Description 4" style="width: 100%;">
    <img src="/images/taylor_green_vortex_Float32sr_shift.gif" alt="Description 5" style="width: 100%;">
    <img src="/images/taylor_green_vortex_Float16_unshift.gif" alt="Description 3" style="width: 100%;">
    <img src="/images/taylor_green_vortex_Float16_shift.gif" alt="Description 4" style="width: 100%;">
    <img src="/images/taylor_green_vortex_Float16sr_shift.gif" alt="Description 5" style="width: 100%;">
</div>

It's clear that the FP16 case doesn't decay at all, so it isn't usable. In a similar manner to Lehman et al., we analyzed the decay in total energy over time as the vortex decays. You can easily see where roundoff error dominates the overall simulation error. The plot below summarizes our results.

<img src="/images/TG_SR.png"
     style="display: block; margin: 0 auto;"
     alt="Taylor Green Vortex Roundoff error comparison"/>

The black dashed line is the analytical solution. As the vortex decays, a given line will depart from the black dashed line when the truncation error dominantes, and the energy is too small to be computed accurately using the selected floating point representation. "Optimized" in this plot means using the DDF-shifting trick discussed above. "Deterministic" means using standard rounding, and "stochastic" means using stochastic rounding.

We drew the following conclusions from this plot:

- We confirm Lehman et al.'s results that DDF shifting improves the performance of deterministically rounded FP32 and FP64 representations.
- DDF shifting also improves the performance when using stochastic rounding floating point representations.
- Stochastic rounding improves the performance of the vanilla LBM algorithm with FP32 and FP64 representations, but worsens the performance of the DDF-shifted LBM algorithm.
- The high roundoff error in FP16 means that it is unacceptable for use in LBM simulations, even with DDF shifting.

## Analysis
We wanted to investigate in more detail the reasons why stochastic rounding didn't improve the DDF-shifted LBM algorithm. David found that because DDF shifting greatly reduces the deterministic bias error, there is minimal improvement when stochastic rounding is employed (i.e., there really isn't any more deterministic bias error to be eliminated). In order to reduce deterministic bias, stochastic rounding adds variance, whic hin this case worsens performance. [See David's analysis here](/docs/SR_error_analysis.pdf) for more details. The results below show the bias/variance differences in the various algorithms.

<img src="/images/bias_variance_SR.png"
     style="display: block; margin: 0 auto;"
     alt="Bias and variance error comparison"/>

## Summary
If you're looking to improve the roundoff error performance of the lattice Boltzmann method, you're better off using DDF-shifting (see [Lehman et al.'s paper](https://arxiv.org/abs/2112.08926)). Stochastic rounding improves the performance of the vanilla LBM algorithm. However, it does so by trading increased variance for reduced bias. DDF shifting greatly reduces this bias alraedy, so the increased variance when using stochastic rounding is not helpful.
