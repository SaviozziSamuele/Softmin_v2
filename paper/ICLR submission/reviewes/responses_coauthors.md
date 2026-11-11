# ICLR 2027 rebuttal: replies to reviewers

## General response

We thank the three reviewers for careful and detailed reviews. This note summarizes the points raised by more than one reviewer and what we will change; the individual replies give the details.

**1. Scope of the barrier result.** Our barrier results are statements about the Freidlin–Wentzell quasipotential: the exponential rate of escape as $\sigma\to0$, at fixed barrier height $h$ and inverse temperature $\beta$. Theorem 3.4 bounds this quasipotential through an explicit channel of paths. Exit times enter only through the conditional Corollary 3.5. "Independently of the barrier height" refers to this exponent. As Reviewer q1kz points out, the limit is taken at fixed $h$, and sub-exponential factors (Eyring–Kramers prefactors) are not controlled and may depend on $h$. We will state this wherever the cap appears.

**2. Corollary 3.5.** Reviewers q1kz and nDsh both note that the exit theorem recalled in Appendix B does not fit the basin $D_-$. The boundary of $D_-$ contains the index-one saddle $\Gamma(x_m)$, and $\Gamma(s)$ is a second attractor, so the corollary says nothing about the passage beyond $\xi_+$. We agree. The quasipotential bounds of Theorem 3.4 are unaffected. We will restate the corollary with a theorem that allows a characteristic boundary, restricted explicitly to exit from $D_-$.

**3. What is specific to $J_\beta$.** Reviewer nDsh shows that the log-sum-exp energy also gives an $O(1/\beta)$ channel cap. We agree, and will say so: the order of the cap comes from Gibbs weighting, which both energies share. What the sign-changing pre-factor of $J_\beta$ adds at the level of the quasipotential is a smaller constant. Along the same channel, the Softmin cap is about 0.40 times the log-sum-exp one at $n=2$ and decreases to $1/e$ times it as $n$ grows. It also changes the post-saddle structure, and the cost of both crossings together stays below the log-sum-exp cost at every barrier height. This agrees with our own ablation (Appendix E.5.4), which already pointed to Gibbs weighting as the main source of the advantage at $\beta=10$.

**4. Cost accounting.**
- Every gradient-based method in our comparison runs the same ensemble size, so the number of gradient evaluations per step is the same for all of them.
- On the transformer, Adam and SGD run as 10 independent chains for the same reason.
- Unsuccessful runs on the well benchmarks are stopped early by two algorithm-agnostic rules: ensemble collapse, or no improvement over 500,000 steps. This explains the Table 12 numbers Reviewer nDsh checked. It also makes the reported cost ratios conservative, since failed baseline runs are charged only the steps they ran.
- The caption of Table 12 is wrong on this point and will be corrected.

**5. Baselines and controls.** In a revised version we will add:
- Simulated Annealing with initial temperature and cooling rate selected jointly by the same expected-running-time criterion used for $\beta$;
- the log-sum-exp drift and a pure-noise control on the well benchmarks;
- Swarm-based Simulated Annealing and Swarm-Based Gradient Descent, in the related work and as baselines;
- basin-hopping on Lennard-Jones clusters.

We have not run these during the discussion period and report no results for them here. Until then, our Simulated Annealing conclusions should be read as conditional on the calibrated schedule.

**6. Presentation.** We will reorganize Section 3 around what is proven unconditionally (quasipotential bounds), what is conditional (exit times) and what is open (exact transition exponent, prefactors, $d>1$). We will also condense the appendix, which Reviewer q1kz fairly describes as reading like research notes.

## Reply to Reviewer MmRZ

We thank the reviewer for the positive assessment and for precise questions.

**W1: cost, and similarity to independent Langevin.** Every gradient-based method we compare, Langevin included, runs the same ensemble of $n$ particles, so all perform the same number of gradient and objective evaluations per step. The Softmin coupling adds only $n$ Gibbs weights, computed from values already evaluated: $O(n)$ scalar operations. On the transformer, Softmin's update costs 2.5 s per step at depth 50, against 2.0 s for Adam (App. E.4.6).

Beyond the double well, independent Langevin dynamics with the same step size and noise fails where Softmin does not:
- **Quadruple well:** Langevin reaches the target in no run from depth gap 10 on; Softmin in every run up to 30.
- **Lennard-Jones (up to 225 dimensions):** Langevin's energy gap grows roughly linearly with cluster size; Softmin's stays nearly flat (Fig. 2).
- **Ackley:** Langevin never reaches the target; Softmin does in every run from $d=40$.
- **Transformer:** Softmin needs 1.4–2.3 times fewer steps than Langevin at every depth.

With $n=1$ the pre-factor is identically 1 and Softmin is exactly Langevin, so the difference comes from the interaction alone: five coupled particles reach the target in every run, where 100 independent Langevin particles never do (Sec. 4.3). We agree that Theorem 2.4 only matches gradient flow; the two methods separate at the barrier (Theorem 3.4).

**W2: the interaction acts only through a scalar.** It changes the sign as well as the magnitude: a particle with $f(x_k)>J_\beta+1/\beta$ performs gradient ascent. Keeping it scalar is deliberate. It makes the dynamics the gradient flow of a single energy $nJ_\beta$, which is what allows Theorems 2.1 and 3.4; consensus- and attraction-based interactions are not gradient flows. It avoids the premature consensus that made our CBO and PSO baselines collapse in the wrong basin (App. E.3). And it has a single parameter, $\beta$. Position-based terms are a natural extension, at the price of the gradient structure.

**Q1: Remark 3.6, made explicit.** We will replace it with
$$\text{Langevin: }\lim_{\sigma\to0}\sigma\log\mathbb E\,\tau=h,\qquad\text{Softmin: }V_\beta\big(\Gamma(z_-),\Gamma(s)\big)\le u_h\le u_c<\frac{n}{e(n-1)\beta}.$$
A particle at height $r$ above the others has Gibbs weight about $e^{-\beta r}/(n-1)$, so its contribution to $nJ_\beta$ along the channel is about $\frac{n}{n-1}re^{-\beta r}$, largest near $r=1/\beta$. Above the switching level $r_c\in\big(\frac1\beta,\frac{n}{(n-1)\beta}\big)$ the pre-factor is negative, and the drift pushes the particle up towards the saddle $s$. So the noise only has to lift it by about $1/\beta$; the drift does the rest. Past $s$, a second crossing of cost at most $u_c-u(h)$ leads into the target basin (Theorem 3.4, part 3). At $\beta=10$, $n=100$ the bound is 0.037, against barriers up to $h=224.8$. This compares an upper bound on a quasipotential with Langevin's exact exponent; exit times follow only under the hypotheses of Corollary 3.5, and the remark will say so.

**Q2: fixed iteration budget.** From the runs behind Fig. 1, we counted how many of the 100 runs reached the target within $10^3$, $10^4$, $10^5$ and $5\cdot10^5$ ensemble steps. All methods use $n=100$ particles, so a step costs the same for each.
- **Langevin** reaches the target in no run at any budget, on every double-well barrier ($h$ from 24.9 to 224.8) and every quadruple-well gap (10 to 30).
- **Simulated Annealing** is fastest at the smallest budget on low barriers: within $10^3$ steps it succeeds in all 100 runs on the two lowest double-well barriers, against 9–12 for Softmin. Its count then stops growing after about $10^4$ steps, once cooling has frozen the dynamics. It ends at 86 runs at $h=99.8$, 1 at $h=156.1$ and 0 at $h=224.8$, and at 17 and 0 for quadruple-well gaps 25 and 30.
- **Softmin** starts slower but keeps progressing. Within $10^4$ steps it succeeds in 50–100 runs on the double well and 33–100 on the quadruple well, within $10^5$ steps in at least 98 runs everywhere, and within $5\cdot10^5$ steps in all of them.

We will add this view as a figure.

**Q3: PL reference.** Polyak (1963) and Łojasiewicz (1963); for the modern use in optimization, Karimi, Nutini & Schmidt (2016). We will cite them in Definition 2.3.

## Reply to Reviewer q1kz

We thank the reviewer for a careful reading of the theory. We agree with part of the criticism and want to be precise about what the paper claims.

**What Theorem 3.4 claims.** It bounds the Freidlin–Wentzell quasipotential $V_\beta$, the exponential rate as $\sigma\to0$ at fixed $h$ and $\beta$, through an explicit channel of paths. It does not claim exit times: the only exit-time statement is the conditional Corollary 3.5, and Remark 3.6 and App. D.2–D.3 compare a quasipotential bound with an exponent. "Independently of the barrier height" refers to this exponent.

**Prefactors.** The reviewer is right that the statement is not uniform in $h$: the limit is taken at fixed $h$, and sub-exponential factors can depend on $h$. Past the switching level the channel drift is of order $\beta r\,e^{-\beta r}$, so the noiseless traversal of the plateau takes a time of order $e^{\beta h}$. For $h>r_c$, the curvature of the post-saddle well at $\Gamma(s)$ is about $\frac{n}{n-1}\beta h\,e^{-\beta h}|f''(s)|$. An Eyring–Kramers prefactor built on these would grow exponentially in $\beta h$. This lies outside the large-deviation statement, which is why the paper does not claim exit times. We will say explicitly, including in the abstract and the contributions, that the cap concerns the exponent, that the limit is taken at fixed $h$, and that prefactors are not controlled.

On the specific step, we could not find an approximation $r\ll e^{\beta r}$ in Lemma 3.3 or its proof. The cap follows from the exact inequality
$$u(r)=\frac{n\,r\,e^{-\beta r}}{n-1+e^{-\beta r}}<\frac{n}{n-1}\,r\,e^{-\beta r}\le\frac{n}{e(n-1)\beta}\qquad(r>0).$$
If the reviewer had another line in mind, we would be grateful for the pointer.

*Illustration, not a claim of the paper:* the $e^{\beta h}$ factor is not forced in the reduced process. Let the active particle move in $W=u\circ r$ with the others held at $z_-$, reflected at $a<z_-$ with $f'<0$ on $[a,z_-)$. Its mean time to reach $b\in(a,z_+]$ is exactly $T(x)=\sigma^{-1}\int_x^b e^{W(y)/\sigma}\int_a^y e^{-W(z)/\sigma}dz\,dy$. Since $W$ rises by at most $u_c$ along the channel, including the post-saddle well,
$$\mathbb E_x\tau_b\le\frac{(b-a)^2}{2\sigma}\,e^{u_c/\sigma}<\frac{(b-a)^2}{2\sigma}\,e^{n/(e(n-1)\beta\sigma)}.$$
The plateau is crossed by diffusion, not by the slow drift. We do not claim this for the full $n$-particle process.

**Corollary 3.5 and the second crossing.** We agree. Corollary 3.5 concerns only the exit from $D_-$. It cannot describe the passage beyond $\xi_+$, because $\Gamma(s)$ is itself a local minimum of $nJ_\beta$ when $r_c<h$. For that stage the paper gives only a quasipotential bound, and App. D.2 states that the transition exponent is not identified; we will move this caveat to the main text. As Reviewer nDsh also notes, the exit theorem of App. B excludes equilibria on $\partial D$, while $\Gamma(x_m)$ is an index-one saddle on $\partial D_-$. We will restate the corollary with a theorem that allows a characteristic boundary (details in our reply to nDsh).

**Presentation.** We take the point. Section 3 will follow one line (reduced potential, unconditional quasipotential bounds, conditional exit-time statement) and will state explicitly what is proven, what is conditional and what is open (exact transition exponent, prefactors, $d>1$). The appendix will be condensed, stating once the calibration of each method with its selection criterion, the caveats and the early-stopping rule.

**Transformer.** We agree that it is inconclusive about reliability, as the paper says. On cost, Adam and SGD run as 10 independent chains like the other baselines, so every method performs the same 10 forward and backward passes per step (Table 6, App. E.4.6); Softmin adds only 10 Gibbs weights. In wall-clock time Softmin costs 2.5 s per step at depth 50, against 2.0 s for Adam and 1.8 s for SGD, and SGD is faster than Softmin from depth 5 on. We will plot gradient evaluations and wall-clock time instead of steps, and present the benchmark as a scalability check, not as evidence of an advantage.

## Reply to Reviewer nDsh

We thank the reviewer for a detailed review; we checked its calculations and agree with most of them.

**Q1: calibration of Simulated Annealing.** Agreed: $\beta$ and $T_0$ were selected under different criteria. At the calibration point SA's escape time decreases monotonically in $T_0$, so selection by escape time or ERT would pick the largest $T_0$ in any grid; we excluded that with the Arrhenius criterion. That concerns what makes a meaningful baseline and does not justify a different objective. We have not re-run SA during the discussion, so we cannot say whether the high-barrier advantages survive. Under geometric cooling, the time spent above any temperature grows only logarithmically in $T_0$, so a larger $T_0$ should shift the barrier at which SA freezes rather than remove it. The question is whether one pair $(T_0,\rho)$ chosen at mid-sweep stays reliable and cheap over barriers spanning three orders of magnitude. We will select $(T_0,\rho)$ jointly by ERT at the same point as $\beta$, without the Arrhenius restriction, extend the sweep until it fails, and report the outcome either way. Until then, our SA conclusions are conditional on the calibrated schedule.

**Baselines.** We agree that Swarm-based Simulated Annealing (Ding et al., 2024) and Swarm-Based Gradient Descent (Lu et al., 2022) are the closest relatives of our method. We will add them to the related work and as baselines, together with basin-hopping on Lennard-Jones. Until then, our Lennard-Jones results compare particle dynamics with each other, not with the state of the art.

**Q2: what $J_\beta$ adds beyond log-sum-exp.** The calculation is correct: the $O(1/\beta)$ cap comes from Gibbs weighting, which both energies share, and we will say so. The sign change adds a smaller constant. From $q(r_c)=0$, $u_c=n(r_c-1/\beta)$ exactly, so $u_c/v_\infty=(\beta r_c-1)/\log\frac{n}{n-1}$: 0.40 at $n=2$, decreasing to $1/e$. More generally $u\le v$ on $r\ge0$, since $u'=nw_1[1-\beta(f-J_\beta)]\le nw_1=v'$. The post-saddle structure cuts both ways: for $h>r_c$, $J_\beta$ creates a shallow well at $\Gamma(s)$ that $F_\beta$ lacks, but the two-stage cost $2u_c-u(h)$ still stays below $v(h)$ for every $n\ge2$ and $h>r_c$. So the gain is a constant factor in the channel bound on the exponent, not a different order in $\beta$. We will add this comparison to Section 3. Empirically, only the $\beta=2$ ablation (App. E.5.4) shows the sign mattering, up to 7 times faster than the absolute-value variant; at $\beta=10$ the variants are close. We will add the drift $-nw_k\nabla f(x_k)$ and a pure-noise control on the wells, with the same seeds, $n$, $\sigma$ and budget.

**Q3: the exit theorem.** The reviewer is right. App. B's theorem excludes equilibria on $\partial D$, but $\Gamma(x_m)\in\partial D_-$ is an index-one saddle. For $h>r_c$ the Hessian there is diagonal, since $f'(z_-)=0$: the active block is $u''(r_c)f'(x_m)^2=-n\beta w_1f'(x_m)^2<0$ and the frozen blocks are $\frac{n}{n-1}f''(z_-)>0$. For $h<r_c$, $x_m=s$ and the active block is $u'(h)f''(s)<0$. We will restate the corollary, restricted to exit from $D_-$, with an exit theorem that allows a characteristic boundary. We are still checking which result fits our setting and its exact hypotheses, and will state them. For a gradient drift such a result gives $\lim_{\sigma\to0}\sigma\log\mathbb E\,\tau_{D_-}=\min_{\partial D_-}nJ_\beta-nJ_\beta(\Gamma(z_-))\le u_h$. Verified: $\Gamma(z_-)$ and $\Gamma(x_m)$ are nondegenerate (Remark C.3 and above). Not established: boundedness of $D_-$ and the equilibria on $\partial D_-$ ($nJ_\beta$ is not confining). Open: whether a cheaper collective saddle exists, which would only lower the exponent. For $h>r_c$, a second saddle of the same height lies on the outer side of $z_-$, so the corollary concerns when $D_-$ is left, not where.

**Q4: early stopping and Table 12.** Yes. Every method stops on ensemble collapse (largest per-coordinate std below $10^{-6}$) or stagnation (no improvement, relative tolerance $10^{-4}$, over 500,000 steps), checked every 1,000 steps. All 1,625 failed SA runs end by collapse at 500,000–521,000 steps (mean 512,900), giving $0.275\times100\times512{,}900\approx1.41\times10^7$, the value in the table. All 4,540 failed Langevin runs stop by stagnation at 549,000 steps. The caption is wrong: a failed run contributes the steps it ran, not the full budget. We will fix it and describe the rule in App. E.3. This makes the cost ratios conservative. The SA stop is absorbing (an uncapped 1.5-million-step check shows no change after step 10,000). The Langevin stop is not, but at the smallest failing barrier ($h\approx24.9$, $\sigma=1$) Kramers' formula gives about $10^8$ time units for the first of 100 independent particles to escape, against a full budget of $10^4$.
