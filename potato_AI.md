## Architecture

You need **256 potatoes** (16×16 grid), each one acting as a calibrated variable resistor representing one weight `w_ij`.

```
        col 1   col 2   col 3  ...  col 16
row 1   🥔w11   🥔w12   🥔w13  ...  🥔w1,16   → Σ current = output_1
row 2   🥔w21   🥔w22   🥔w23  ...  🥔w2,16   → Σ current = output_2
row 3   🥔w31   🥔w32   🥔w33  ...  🥔w3,16   → Σ current = output_3
 ...
row 16  🥔w16,1 ...                🥔w16,16  → Σ current = output_16

input voltages x1...x16 applied down each COLUMN
```

Each potato sits at a row/column intersection with two electrodes (zinc + copper). Input voltage `x_j` is applied along column `j`. Current flows through each potato proportional to `x_j / R_ij`. All 16 potatoes in a *row* dump their current into a shared collector wire — Kirchhoff's current law sums them automatically. That summed current at row `i` **is** `Σ_j (x_j · w_ij)`, i.e., row `i` of `Wx`. That's a full matrix-vector multiply, no transistors doing the arithmetic.

## Calibrating weights via ripeness/size

Since you specifically want to encode weight values through spud selection, here's a rough spec:

| Weight magnitude | Potato size | Ripeness | Approx. resistance |
|---|---|---|---|
| High (~1.0) | Large (Russet, 300g+) | Very ripe/soft | ~200-400 Ω |
| Medium (~0.5) | Medium (150g) | Moderately ripe | ~600-900 Ω |
| Low (~0.1) | Small (new potato, 50g) | Firm/fresh | ~1500-2500 Ω |
| Negative weight | — | — | **Not achievable** — potatoes can't source negative resistance. You'd need a second parallel row wired with reversed electrode polarity to fake signed weights (differential pair trick, same as real analog crossbar chips do) |

## The hard engineering problems at 16×16 scale

1. **256 potatoes is a lot of decaying organic matter.** Each one is basically a slow chemical reaction consuming its own electrodes (zinc oxidizes over hours). Expect calibration drift within 4-8 hours — your "weights" rot in real time. This is either a huge bug or, if you're framing it as art, a beautiful metaphor for catastrophic forgetting.

2. **Signal is tiny and noisy.** 256 potatoes' worth of current summed per row is still microamps to low milliamps. You'd want an op-amp transimpedance amplifier on each of the 16 row outputs just to get a readable signal — otherwise your "clock" and any readout electronics won't register anything.

3. **No nonlinearity.** This whole rig computes exactly one linear transform, `Wx`. To make it an actual neural net layer you'd still need a digital or analog nonlinearity (even a diode clipper for a crude ReLU) after each row's output.

4. **Reprogramming = grocery shopping.** Changing weights means swapping potatoes for ones of different size/ripeness. Backprop, in this world, is you standing in the produce aisle.

5. **Physical footprint.** 256 potatoes with two electrodes each is ~512 wire connections. This is less "chip" and more "science fair table that ate a farm."
