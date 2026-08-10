
### A. Token geometry

For token (i), with frequency (f_i) and vocabulary index (k_i),

[
\hat f_i=\frac{f_i}{f_{\max}},
\qquad
\hat k_i=\frac{k_i}{V-1}.
]

The source constructs

[
p_i=
\hat f_i
\begin{bmatrix}
\cos(2\pi\hat k_i)\
\sin(2\pi\hat k_i)
\end{bmatrix}
]

and

[
q_i=
\hat k_i
\begin{bmatrix}
\cos(2\pi\hat f_i)\
\sin(2\pi\hat f_i)
\end{bmatrix}.
]

This is explicitly how the original geometry is parameterized. 

For (p,q), the original geometric triple is

[
\rho =
\left(
1-\frac{||p|-|q||}{|p|+|q|}
\right)
(1-|\cos\phi|)
]

with

[
\cos\phi=\frac{p^\top q}{|p||q|},
]

while

[
\sigma=\frac14\sum_{j=1}^{4}|T_{j+1}-T_j|
]

and

[
\theta=\operatorname{atan2}(\Delta y,\Delta x)\pmod{\pi}.
]

That is the actual source definition. 

**This geometry must be frozen identically in the new model.**

---

# 2. The original synaptic plot/operator

The candidate-candidate kernel is

[
K_{ij}
======

e^{-\gamma(\sigma_i-\sigma_j)^2}
\frac{1+\cos(\theta_i-\theta_j)}2
e^{-\lambda(\rho_i-\rho_j)^2}.
]

The source then removes self-connections and normalizes each row:

[
W_{ij}
======

\frac{
K_{ij}\mathbf 1_{i\ne j}
}{
\sum_jK_{ij}\mathbf 1_{i\ne j}
}.
]

This is explicitly a **SUM-normalized synaptic operator**. 

Then:

[
z_{\rm syn}=Wz
]

and the source applies layer normalization:

[
\operatorname{LN}(z)
====================

\frac{z-\mu(z)}
{\sigma(z)+\epsilon}.
]



So the new neural version should retain:

[
\boxed{
z_{\rm syn}
===========

\operatorname{LN}(Wz)
}
]

rather than replacing it with a generic linear layer.

---

# 3. The transitive SUM must remain

The source doesn't simply use the candidate geometry.

For context (w_1,w_2) and candidate (c):

[
p_T=
\frac14p_1+
\frac12p_2+
\frac14p_c
]

and

[
q_T=
\frac14q_1+
\frac12q_2+
\frac14q_c.
]

Then the same Thébault transformation produces

[
(\rho_T,\theta_T,\sigma_T).
]



That is an important SUM structure:

[
\boxed{
T
=

\frac14G(w_1)
+
\frac12G(w_2)
+
\frac14G(c)
}
]

So the neural architecture should learn around this operation, **not replace it with an arbitrary context embedding**.

---

# 4. The actual original enrichment equation

The source constructs a large raw logit sum. The relevant structural part is

[
z_0
===

b_{\tau}
+
b_{\rm influence}
+
b_{\rm mandate}
+
b_{\rm CoT}
+
b_{\rm PDN}
+
b_{\rm chunk}
+
b_{\rm echo}
+
b_{\rm MRV}
+
b_{\rm orbit}
+
b_{\rm composition}
+
b_{\rm graph}
+
b_{\rm geometry}
+
b_{\rm punctuation}
+
\log P_{\rm LM}.
]

The ordering is reversed in the source, but the additions are commutative, so the scalar value is unchanged. 

The key point for the new model is:

[
\boxed{
z_0=\sum_r b_r
}
]

not

[
z_0=\operatorname{MLP}(x).
]

The MLP can learn the **coefficients of the terms**, but the structural SUM remains.

---

# 5. Influence-space term

This was another thing my previous code omitted.

The original candidate kernel is transformed by:

[
W_0
\overset{\operatorname{rank/log}}{\longrightarrow}
W_1
\overset{\exp(\alpha W)}{\longrightarrow}
Y.
]

The source defines

[
\ell_{ij}
=========

-\log(1+\operatorname{rank}(W_{ij}))
]

then shifts the result and applies

[
Y=e^{\alpha W_1}.
]

The candidate influence is

[
I_i
===

\sum_jY_{ij}-Y_{ii}.
]



That is a genuine **matrix SUM + matrix exponential** component.

For the new network:

[
\boxed{
I=\left(e^{\alpha\mathcal R(W)}\right)\mathbf1
----------------------------------------------

\operatorname{diag}
\left(e^{\alpha\mathcal R(W)}\right)
}
]

where (\mathcal R) is the rank/log transform.

---

# 6. Then comes CSNS

The source combines:

[
z_{\rm enriched}
================

z_0
+
\omega_{\rm trans}b_{\rm trans}
+
\omega_{\rm syn}z_{\rm syn}.
]



Therefore the trainable version should have:

[
\boxed{
z_e
===

z_0+
\omega_tT+
\omega_sWz_0
}
]

with

[
\omega_s,\omega_t>0
]

learned through softplus parameters.

---

# 7. The critical DNN pipeline

This is where the previous code diverged.

The actual source explicitly uses:

[
\boxed{
\sigma
\rightarrow
\theta
\rightarrow
\operatorname{ReLU}_{\theta}
\rightarrow
\rho
\rightarrow
T
\rightarrow
\operatorname{simplex}
}
]

not a generic SUM/DIAG stack. 

### Sigma

[
s_\sigma
========

0.7+
0.3
\frac{\sigma}
{\max\sigma+\epsilon}.
]

Then

[
z_1=D_\sigma z_e.
]

where

[
D_\sigma=\operatorname{diag}(s_\sigma).
]

### Theta

[
s_\theta
========

\frac12(1+\cos\theta).
]

The next activation is

[
z_2
===

\operatorname{signedPower}
\left(
D_\theta z_1+0.3z_e,
1.5
\right).
]



### Orientation ReLU

Define

[
g_i=
\operatorname{ReLU}
\left(
s_{\theta,i}
-\overline{s_\theta}
\right)
]

and

[
\hat g_i=
\frac{g_i}{\max_jg_j+\epsilon}.
]

Then:

[
z_{2b}
======

\hat g\odot z_2
+
(1-\hat g)\odot\operatorname{ReLU}(z_2).
]

This is the actual plotted orientation gate. 

### Rho

The source uses

[
r_i
===

1+
0.5
\operatorname{clip}
\left(
\frac{\rho_i-\mu_\rho}
{\sigma_\rho+\epsilon},
-2.5,
2.5
\right).
]

Then:

[
z_3
===

\operatorname{signedPower}
(
D_\rho z_{2b},
2
).
]



---

# 8. Temperature is geometric, not ordinary softmax temperature

This is another major difference.

The source does **not** simply calculate

[
z/T.
]

It calculates

[
w_i(T)
======

\exp
\left[
-\frac{\lambda_T(\rho_i-\bar\rho)^2}
{\max(T,0.1)}
\right]
]

and therefore

[
z_i'
====

z_{3,i}w_i(T).
]



So:

[
\boxed{
T\text{ changes the geometry of the distribution}
}
]

rather than merely flattening logits.

---

# 9. And the "probability" plot is not softmax

This was the biggest mathematical mistake in the ZIP.

The source does:

[
x'=x-\min_i x_i
]

then

[
x_+ =
\frac{x'^2}{|x'|+\epsilon}
]

with the source's smooth-power function, followed by

[
P_i=
\frac{\max(x_{+,i},\epsilon)}
{\sum_j\max(x_{+,j},\epsilon)}.
]



Therefore the actual plotted probability is:

[
\boxed{
P_i
===

\frac{
\operatorname{SPR}(z_i-\min z)
}{
\sum_j\operatorname{SPR}(z_j-\min z)
}
}
]

not

[
\operatorname{softmax}(z)_i.
]

That means the new model must preserve this projection if the plots are supposed to match.

---

# 10. The exact new mathematical model

So I would define the trainable network as:

[
\boxed{
P_\theta(c|w_1,w_2)
===================

\Pi_1
\left[
G_T
\left(
D_\rho
\left|
R_\theta
\left|
D_\theta
D_\sigma
z_e
+
0.3z_e
\right|^{1.5}
\right|^{2}
\right)
\right]
}
]

where:

[
z_e
===

z_0
+
\omega_tT_c
+
\omega_sWz_0
]

and

[
z_0
===

\sum_r
\alpha_r
\phi_r(w_1,w_2,c).
]

The (\phi_r)'s are the **same features the old algorithm plotted**.

That gives us the new neural parameterization:

[
\boxed{
\theta=
{
\alpha_r,\omega_s,\omega_t,
\lambda_T,
\text{diagonal gate parameters},
\text{context parameters}
}
}
]
