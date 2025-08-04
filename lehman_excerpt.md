The following is an excerpt from Moritz Lehman's paper, it can be found here [here](https://arxiv.org/pdf/2112.08926).


### 2.2 DDF-shifting

To achieve maximum accuracy, it is essential not to work with the density distribution functions (DDFs) \( f_i \) directly, but with shifted \( f_i^{\text{shifted}} := f_i - w_i \) instead [50, 57, 61, 83, 89].
\( w_i = f_i^{\text{eq}}(\rho = 1, \vec{u} = 0) \) are the lattice weights and \( \rho \) and \( \vec{u} \) are the local fluid density and velocity. This requires a small change in the equilibrium DDF computation:

\[
f_i^{\text{eq-shifted}}(\rho, \vec{u}) := f_i^{\text{eq}}(\rho, \vec{u}) - w_i =
\tag{1}
\]

\[
= w_i \cdot \rho \cdot \left( \frac{(\vec{u} \cdot \vec{c}_i)^2}{2c^4} + \frac{\vec{u} \cdot \vec{c}_i}{c^2} + 1 - \frac{\vec{u} \cdot \vec{u}}{2c^2} \right) - w_i =
\tag{2}
\]

\[
= w_i \cdot \rho \cdot \left( \frac{(\vec{u} \cdot \vec{c}_i)^2}{2c^4} + \frac{\vec{u} \cdot \vec{c}_i}{c^2} - \frac{\vec{u} \cdot \vec{u}}{2c^2} \right) + w_i \cdot (\rho - 1)
\tag{3}
\]

and density summation:

\[
\rho = \sum_i \left( f_i^{\text{shifted}} + w_i \right) = \left( \sum_i f_i^{\text{shifted}} \right) + 1
\tag{4}
\]

We emphasize that it is key to choose equation (3) exactly as presented without changing the order of operations\(^1\), otherwise the accuracy may not be enhanced at all [57, 61, 89]. With this exact order, the round-off error due to different sums is minimized. This offers a large benefit, most prominently on FP16 accuracy, by substantially reducing numerical loss of significance at no additional computational cost. Since it is also beneficial for regular FP32 accuracy, it is already widely used in LBM codes such as our FluidX3D [6–10], OpenLB [63–66], ESPResSo [22–24], Palabos [67–71] and some versions of waLBerla [50]. In the appendix in section 8.2 we provide the entire equilibrium method with and without DDF-shifting for comparison and in section 8.3 we clarify our notation.

We also recommend doing the summation of the DDFs in alternating ‘+’ and ‘–’ order during computation of the velocity \( \vec{u} \) to further reduce numerical loss of significance, for example \( u_x = (f_1 - f_2 + f_7 - f_8 + f_9 - f_{10} + f_{13} - f_{14} + f_{15} - f_{16}) / \rho \) for the x-component in D3Q19.
