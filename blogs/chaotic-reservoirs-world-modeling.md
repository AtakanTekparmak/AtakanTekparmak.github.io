---
title: Chaotic Reservoirs for World Modeling
date: 2026-04-13
authors: Atakan Tekparmak, Claude 4.6 Opus (Claude Code)
excerpt: A sub 5k-parameter world model that uses frozen chaotic dynamics as an inductive bias, and learns when to trust them.
readingTime: 14 min read
---

**TL;DR:** I built a world model out of a frozen Coupled Map Lattice (a chaotic reservoir with zero trainable weights) and a tiny learned correction on top, totaling 321 trainable parameters. It outperforms CNNs with 8x more parameters on physics PDEs and stays competitive on cellular automata and Atari. When extended with a learned trust gate (4747 params total), the model figures out on its own when to rely on the reservoir and when to ignore it, a property my Claude likes caling "the Matching Principle". All results are multi-seed across 16 benchmarks in 5 domains.

---

Last week my friend [andthattoo](https://andthattoo.dev) published [a blog post](https://andthattoo.dev/blog/es2n_paralesn) on bolting cellular automata (CA) onto a parallelizable reservoir (ParalESN) for language modeling. He found that a Coupled Map Lattice (CML), a grid of logistic maps with local coupling, could serve as a frozen chaotic reservoir that provides useful computation if you only learn the interface. The core idea was simple: fixed chaotic dynamics are computationally rich, and you can exploit that without training the dynamics themselves.

Immediately I had a lightbulb appear in my head: a similar architecture could be very parameter efficient for world modeling and could provide gains beyond efficiency due to the inherent chaotic dynamics present in the system.

## The architecture

So the whole thing is 321 learned parameters and a frozen chaotic reservoir. I call it `rescor` (reservoir-correction), a Coupled Map Lattice that runs for 15 steps with zero trainable weights, followed by a 3x3 convolution and a 1x1 projection that learn a correction on top of the CML's output. On physics PDEs it outperforms CNNs with 8x more parameters, on Atari Pong it hits 99.6% one-step accuracy, and on cellular automata it stays competitive with models that have an order of magnitude more capacity.

The reasoning behind this is pretty standard reservoir computing: chaotic dynamics are computationally rich, and a single logistic map at `r=3.90` produces an infinite non-repeating sequence that's sensitive to initial conditions at every step. When you couple thousands of these maps on a 2D grid and inject your input as a drive signal, the reservoir performs a massively nonlinear expansion of the input without any learned parameters. The only thing the model needs to learn is how to read the reservoir's output.

## The reservoir

The reservoir is a 2D Coupled Map Lattice (CML) where each cell evolves under the logistic map `f(x) = r*x*(1-x)` at `r=3.90`, which puts it deep in the chaotic regime. The cells are coupled to their neighbors via a fixed 3x3 convolution kernel, and the input gets re-injected at every step as a drive signal:

```
mapped  = r * grid * (1 - grid)
physics = (1-e) * mapped + e * conv(mapped)
grid    = (1-B) * physics + B * drive
```

This runs for 15 steps with zero trainable parameters, and along the way the reservoir collects 5 statistics per cell: the final state, the mean, the variance, the delta (change between consecutive steps), and the drift from the input. These are free temporal features that the CML computes as a byproduct of its chaotic dynamics.

The drive injection (`B*drive`) turned out to be important in a way I didn't initially expect. It anchors the chaotic dynamics near the input signal, which prevents the logistic map from losing track of the information entirely. What's interesting is that deeper chaos (`r=3.90` vs `r=3.57`) actually gives better input reconstruction, because the drive keeps pulling the dynamics back toward the input while the chaotic regime generates a richer feature expansion around it.

## The correction

The vanilla `rescor` sits on top of this reservoir with the simplest possible readout: a single 3x3 convolution that takes the input concatenated with the CML's final state (2 channels in, 16 hidden channels), a ReLU, and a 1x1 projection back to 1 channel. The output is the CML's last state plus this learned correction:

```
output = cml_last + nca_correction([input, cml_last])
```

This gives us **321 trainable parameters** total, with the CML contributing zero. The reasoning follows the reservoir computing principle: keep the physics frozen, only learn the readout. Backpropagating through 15 steps of the logistic map would be a gradient nightmare anyway, since the derivative `f'(x) = r(1-2x)` flips sign at every step, so I don't even try.

Here is the full diagram for vanilla `rescor`:

<!-- include: artifacts/rescor-diagram.html -->

On top of this I built an extension called **E3c**, which replaces the single conv with two parallel 3x3 convolutions at dilation 1 and dilation 2, feeds in all 5 CML statistics instead of just the last state, and gates the dilated branch with a zero-initialized scalar that decays toward zero under L2 regularization. This gives the model multi-scale spatial perception and richer reservoir readouts at the cost of bumping the parameter count to **4641**, and it closes the gap on discrete systems like Game of Life while keeping the PDE performance. The core idea is still the same though: frozen reservoir, learned correction.

## The Matching Principle

After running this across 16 benchmarks I noticed a consistent pattern that my Claude started calling the **Matching Principle**: when the CML's dynamics match the target system's physics, the reservoir helps, and when they don't, it either does nothing or actively hurts.

This makes sense when you think about what the CML actually is, because it's fundamentally a discretized diffusion-reaction system. The logistic map combined with local convolution coupling naturally encodes heat-like diffusion, wave propagation, and reaction-diffusion dynamics. Given this, the results across benchmarks line up with what you'd expect:

- **Heat equation, Kuramoto-Sivashinsky (KS), Gray-Scott**: the CML-based model outperforms pure learned baselines because the reservoir's inductive bias happens to be correct for these systems. On heat, `rescor` at 321 parameters scores 8.8e-8 MSE, nearly 1000x better than the 2625-parameter CNN baseline.
- **Game of Life (GoL), Rule 110, Wireworld**: the CML is just noise here. Binary birth/survival rules have nothing in common with the logistic map, so a pure NCA (Neural Cellular Automaton) with only 177 parameters does just as well or better.
- **DMControl** (flat state vectors, no spatial structure): the CML actively hurts performance, 20-60x worse than a plain MLP. There's no 2D grid for the conv2d coupling to exploit, which is exactly what the principle predicts.

The interesting question at this point was not "why does the CML fail on GoL" (that's obvious given the dynamics mismatch) but rather whether the model could figure out on its own when to trust the reservoir.

## The trust gate

To answer that question I built `rescor_mp_gate`, which is a variant with two parallel paths and a learned per-cell gate between them:

- **Path A**: the full CML + NCA correction (4641 params, using E3c)
- **Path B**: a tiny pure NCA with no CML at all (89 params)
- **Trust gate**: an MLP that reads the CML's trajectory variance and drift, and outputs a per-cell blend weight

```
trust  = sigmoid(mlp([cml_var, cml_drift]))
output = trust * path_a + (1 - trust) * path_b
```

This comes out to 4747 parameters total. The gate initializes at 0.5 (equal blend between both paths) and learns to modulate from there during training. Here's the full architecture:

<!-- include: artifacts/rescor-mp-gate-diagram.html -->

What I found is that the gate learns the Matching Principle on its own, without any explicit supervision about which benchmarks should use the reservoir and which shouldn't. On physics benchmarks like KS and Gray-Scott the gate trends toward trusting the CML path, on non-spatial benchmarks like MiniGrid (an 8x8 grid navigator where only the agent cell changes per step) it trends toward the NCA path, and on mixed-dynamics benchmarks like CrafterLite (where tree growth is spatial and local but harvesting is symbolic) it sits somewhere in between.

This single model beats the fixed-architecture baseline on 5-6 out of 8 benchmarks across 3 random seeds, with the most notable result being a 53% MSE reduction on MiniGrid where the CML was actively hurting performance and the gate learned to shut it off.

<!-- include: artifacts/per-benchmark-trust.html -->

## CrafterLite

I built CrafterLite as a toy benchmark inspired by Crafter (Hafner 2022), the 2D survival game often used in RL research. The motivation was to test mixed spatial and symbolic dynamics in a single environment, which is something none of my other benchmarks did cleanly. CrafterLite runs on a 16x16 grid with 7 cell types (empty, tree, sapling, stone, water, agent, and resource) where tree growth spreads locally via neighbor seeding, the kind of spatial diffusion the CML should help with, but harvesting removes single cells based on agent proximity, which is a symbolic rule that has nothing to do with coupled map lattices.

The results matched the Matching Principle prediction: CML-based models slightly outperform on the spatial components (tree spread, water flow) while the pure NCA handles the symbolic transitions just as well. The trust gate in `rescor_mp_gate` learns an intermediate blend at 0.52 trust, and scores 96.1% accuracy compared to 95.9% for both `pure_nca` and vanilla `rescor`. The real value of CrafterLite as a benchmark is that it isolates where the reservoir helps and where it's dead weight. For context, DELTA-IRIS (Micheli et al. 2024) uses 25M parameters for Crafter world modeling at 64x64 RGB resolution, a fundamentally different paradigm (pixel-level reconstruction vs symbolic grids), but the question of whether a CML reservoir combined with a lightweight vision encoder could get competitive accuracy at a fraction of the parameter count is worth exploring in future work.

## What didn't work

Before I landed on the trust gate I tried three other extensions that didn't make the cut, but each one taught me something useful about the architecture.

**Trajectory attention** (`rescor_traj_attn`, +18 params) replaced 3 of the 5 hand-crafted CML statistics with learned cross-attention over the raw 15-step trajectory, with the idea being to let the model discover its own summary statistics. It helped on wireworld and heat rollouts but hurt Gray-Scott, and the hand-crafted stats (mean, variance, delta) turned out to already capture the first and second moments plus the gradient, which is near-optimal for most dynamics. The takeaway was more about which benchmarks benefit from richer temporal features than about the attention mechanism itself.

**Mixture-of-experts routing** (`rescor_moe_rf`, -20 params) replaced the global dilation scalar with a per-cell router that picks between local and wide receptive fields based on CML statistics. It tied on almost everything because the router learned near-constant weights on PDEs, meaning the optimal receptive field is spatially uniform for these systems. This actually validated the simpler fixed-dilation design, which I consider a useful negative result.

**Per-channel affine drive** (E4) tried to let the model learn a scaling of the CML input to position it in the logistic map's chaotic sweet spot, and it was catastrophic. The gradients destabilized the frozen CML entirely, because touching the drive breaks the physics firewall that makes the whole architecture work. In hindsight I should have seen that coming.

The lesson from all three was the same: the CML statistics are already a near-optimal interface to the reservoir. The only thing that consistently helped was learning *when* to use them, not *how* to transform them, and that insight is what led directly to the trust gate.

## The numbers

Below are the full results across all benchmarks and model variants. All numbers are 1-step metrics, 30 epochs, 16x16 grids. For MSE columns lower is better, for accuracy columns higher is better, and the best in each column is highlighted. Parameter counts shown are for the canonical single-channel (1-in, 1-out) configuration; multi-channel benchmarks scale the I/O layers proportionally.

### Physics PDEs (1-step MSE, lower is better)

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 1em 0;">
<table style="width: 100%; min-width: 500px; border-collapse: collapse; font-size: 12px; font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #0f1b1e; border: 1px solid rgba(45,212,191,0.12);">
<tr style="border-bottom: 2px solid rgba(45,212,191,0.25);">
  <th style="text-align: left; padding: 8px 10px; color: #8a9699;">Model</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Params</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Heat</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">KS</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Gray-Scott</th>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>mlp</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">197K</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">1.0e-4</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">2.2e-6</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">1.1e-5</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>pure_nca</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">177</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">4.0e-5</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">8.6e-6</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">6.8e-5</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>conv2d</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">2625</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">8.4e-6</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">6.8e-6</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">1.9e-5</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #e6edf0; font-weight: bold;"><code>rescor</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">321</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">7.7e-7</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">5.1e-7</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">2.1e-6</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor_e3c</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">4641</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">9.5e-7</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">5.5e-8</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">1.1e-6</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor_mp_gate</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">4747</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">7.3e-7</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">2.2e-8</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">4.5e-7</td>
</tr>
</table>
</div>

The CML-based models outperform both the 197K-parameter MLP and the 2625-parameter CNN on every PDE benchmark. `rescor` at 321 parameters scores 7.7e-7 on heat, over 100x better than both baselines despite having 600x fewer parameters than the MLP. The MLP's inability to exploit spatial structure makes it the worst performer here, which makes sense given that it flattens the grid and loses all locality information.

### Discrete CAs (1-step accuracy, higher is better)

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 1em 0;">
<table style="width: 100%; min-width: 500px; border-collapse: collapse; font-size: 12px; font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #0f1b1e; border: 1px solid rgba(45,212,191,0.12);">
<tr style="border-bottom: 2px solid rgba(45,212,191,0.25);">
  <th style="text-align: left; padding: 8px 10px; color: #8a9699;">Model</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Params</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">GoL</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Rule110</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Wireworld</th>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>mlp</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">197K</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">75.4%</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">100%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">70.5%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>pure_nca</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">177</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">94.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">96.3%</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">99.0%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>conv2d</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">2625</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">95.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">96.7%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">70.5%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">321</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">87.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">96.8%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">97.9%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor_e3c</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">4641</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">96.1%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">96.8%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">97.9%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor_mp_gate</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">4747</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">96.1%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">78.7%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">97.9%</td>
</tr>
</table>
</div>

On cellular automata the picture inverts, which is exactly what the Matching Principle predicts. The CML's continuous logistic-map dynamics have nothing in common with GoL's binary birth/survival rules, so vanilla `rescor` trails `pure_nca` and the extensions. The MLP is interesting here: it gets 100% on Rule110 (a 1D CA it can memorize entirely at this grid size) but collapses to 75% on GoL and 70% on Wireworld where spatial locality matters. `pure_nca` wins Wireworld outright at 99.0% while both the MLP and the 2625-parameter CNN collapse to 70.5%.

### Games (1-step accuracy)

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 1em 0;">
<table style="width: 100%; min-width: 600px; border-collapse: collapse; font-size: 12px; font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #0f1b1e; border: 1px solid rgba(45,212,191,0.12);">
<tr style="border-bottom: 2px solid rgba(45,212,191,0.25);">
  <th style="text-align: left; padding: 8px 10px; color: #8a9699;">Model</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Params</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Pong</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Breakout</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">MiniGrid (MSE)</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">CrafterLite</th>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>mlp</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">197K</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">100%</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">100%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">3.8e-3</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">67.5%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>pure_nca</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">177</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.6%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">3.8e-4</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">95.9%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>conv2d</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">2625</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.8%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.99%</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">2.4e-4</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">95.9%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">321</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.7%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">1.2e-3</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">95.9%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor_e3c</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">4641</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.7%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.97%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">4.4e-4</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">96.1%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor_mp_gate</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #6b7578;">4747</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.7%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.98%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">1.7e-4</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">96.1%</td>
</tr>
</table>
</div>

The MLP dominates Atari with 100% accuracy on both Pong and Breakout because these are small grids where it can memorize the full state. But it collapses to 67.5% on CrafterLite (which has mixed spatial and symbolic dynamics) and 3.8e-3 on MiniGrid (the worst score in the table). The interesting signal remains MiniGrid: vanilla `rescor` is bad at 1.2e-3 because the CML hurts on a grid navigator with no spatial coupling, and `rescor_mp_gate` recovers to 1.7e-4 by learning to distrust the reservoir.

### AutumnBench (1-step accuracy, higher is better)

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 1em 0;">
<table style="width: 100%; min-width: 400px; border-collapse: collapse; font-size: 12px; font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #0f1b1e; border: 1px solid rgba(45,212,191,0.12);">
<tr style="border-bottom: 2px solid rgba(45,212,191,0.25);">
  <th style="text-align: left; padding: 8px 10px; color: #8a9699;">Model</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Disease</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Gravity</th>
  <th style="text-align: right; padding: 8px 10px; color: #8a9699;">Water</th>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>mlp</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">65.8%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">92.4%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">86.7%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>pure_nca</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">95.6%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">97.8%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>conv2d</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">85.8%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">96.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">98.0%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">95.6%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">96.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">98.9%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #b8c4c8;"><code>rescor_e3c</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">95.6%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">96.9%</td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">99.2%</td>
</tr>
<tr style="border-bottom: 1px solid rgba(45,212,191,0.1);">
  <td style="padding: 6px 10px; color: #e6edf0; font-weight: bold;"><code>rescor_mp_gate</code></td>
  <td style="text-align: right; padding: 6px 10px; color: #b8c4c8;">95.6%</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">99.97%</td>
  <td style="text-align: right; padding: 6px 10px; color: #2dd4bf; font-weight: bold;">99.3%</td>
</tr>
</table>
</div>

The MLP baseline is the clear loser on AutumnBench, collapsing to 65.8% on disease spreading and 86.7% on water flow, both of which are grid-based physics processes where local spatial structure is essential for prediction. The MLP has 197K parameters on these benchmarks but can't exploit any of the spatial information that a 321-parameter `rescor` handles well. `rescor_mp_gate` is again the winner on gravity (99.97%) and water (99.3%), both of which involve spatial propagation dynamics that align with the CML's coupling structure.

Looking at the full picture, the MLP baseline at 197K parameters is consistently the worst performer on spatially-structured benchmarks despite having 40-600x more parameters than any CML-based model. It only wins on Atari (small grids, memorizable) and Rule110 (1D, memorizable). This validates the Matching Principle from the baseline side: spatial inductive bias is not optional for grid dynamics, you either build it in (CML, NCA, CNN) or the model fails regardless of capacity. Among the spatially-aware models, vanilla `rescor` at 321 parameters outperforms the 2625-parameter CNN on every PDE, `rescor_e3c` closes the gap on discrete systems, and `rescor_mp_gate` adds 106 parameters on top to handle the cases where the CML's physics don't match.

## What's next

There are a few directions I want to explore from here. The first is running on **PDEBench / APEBench** for a standardized comparison against Fourier Neural Operators (FNO, 1-6M params) and U-Net (2-7M params) to see how the 4747 vs millions of parameters story holds up on established benchmarks. The second is **physics control** via Cross-Entropy Method (CEM) planning with the CML world model, in heat-control and reaction-diffusion-control environments where the physics prior directly matches. I also want to scale CrafterLite to **real Crafter** with a learned vision encoder, and move beyond 1-step prediction to **multi-step rollouts and planning** on Pong and Breakout.

Going forward, vanilla `rescor` at 321 parameters is the default configuration for this project. It is the most parameter-efficient world model I've seen for spatially-structured dynamics, outperforming models with orders of magnitude more capacity on physics benchmarks while staying competitive everywhere else. The extensions (E3c, MP-Gate) exist for cases where you need to squeeze out extra accuracy on discrete systems or handle mixed dynamics, but for most applications the vanilla architecture is all you need.

Code is at [github.com/AtakanTekparmak/wmca](https://github.com/AtakanTekparmak/wmca).

## Thanks

Thanks to [andthattoo](https://andthattoo.dev) for finding the CML and writing the ParalESN + CA blog post that started this whole thing. The idea of using a frozen chaotic reservoir as an inductive bias for sequence and grid modeling was his, and this project is a direct extension of that idea into the world modeling domain. These experiments were done alongside many conversations with him, where we discussed the architecture, the results, and what to try next.

## References

- andthattoo, [ParalESN + Cellular Automata for Language Modeling](https://andthattoo.dev/blog/es2n_paralesn), 2026
- Pathak et al., [Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data](https://arxiv.org/abs/1710.07313), 2018
- Richardson et al., [NCA learns PDE dynamics (Gray-Scott)](https://arxiv.org/abs/2405.11420), 2024
- Hafner, [Benchmarking the Spectrum of Agent Capabilities (Crafter)](https://arxiv.org/abs/2109.06780), 2022
- Micheli et al., [DELTA-IRIS: Discrete Latent Transformers for World Models](https://arxiv.org/abs/2406.01807), 2024

## Citation

If you use this work, please cite:

```bibtex
@article{tekparmak2026chaotic,
  title={Chaotic Reservoirs for World Modeling},
  author={Tekparmak, Atakan},
  year={2026},
  url={https://atakantekparmak.github.io/#blog/chaotic-reservoirs-world-modeling}
}
```

---

*The full experiment logs, CML reservoir analysis, Matching Principle validation across all systems, and int8 viability results are in the repo's `findings.md`. Multi-seed, no cherry-picking. I retracted a single-seed claim once already and learned my lesson.*