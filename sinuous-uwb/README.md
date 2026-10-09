# Sinuous ultra-wideband antenna (2–10 GHz)

![preview](sinuous_preview.png)

A 4-arm sinuous antenna built from the standard design equations, simulated in Ansys HFSS through PyAEDT. Arms and substrate only (no balun or cavity yet), matching the group's reference model.

| File | What it does | Needs Ansys? |
|---|---|---|
| `run_sinuous.py` | Builds the HFSS model, solves, exports everything, makes the plots | Yes |
| `sinuous_geometry.py` | The antenna shape (design equations). `python sinuous_geometry.py` makes `sinuous_preview.png` | No |
| `rf_post.py` | Plots + `summary.md` from saved data. `python rf_post.py results/<run>` | No |

## Current design

| Setting | Value |
|---|---|
| Band | 2–10 GHz |
| Arms | 4, alpha 45°, delta 22.5° (self-complementary), tau 0.8 |
| Size | 70 mm arm diameter, 80 mm board |
| Substrate | Taconic TLY, er 2.2, tan d 0.0009, 1.575 mm |
| Feed | 4 lumped ports at the centre, 100 Ω each = 200 Ω per opposite-arm pair |

## Run it (in this order)

At the top of `run_sinuous.py`, set `MODE`:

1. **`"geometry"`**: builds the model and stops (about 1 minute). Nothing is solved.
2. **`"smoke"`**: coarse solve, mesh refined at 6 GHz (about 20 minutes).
3. **`"full"`**: accurate solve, mesh refined at 3, 6 and 9 GHz. Run it overnight.

From the VM terminal:
```bash
cd ~/Research-RF-Antennas/sinuous-uwb
~/pythoncode/.venv/bin/python run_sinuous.py
```

Every run makes its own folder `results/<date>_<mode>/`. **Open `summary.md` there first.** If a run stops with an error, the full message is in `log.txt`.

## Main result: S11 with the 90° feed

`00_S11_quadrature_feed.png` is the S11 with opposite arm pairs (1-3 and 2-4) fed differentially and 90° apart, all four arms driven together. HFSS solves each arm separately; `rf_post.py` combines them. Inside Ansys the same curve is `dB(ActiveS(P1:1))` after setting the sources to 0/90/180/270°.

| Run | Result |
|---|---|
| Oct 8, smoke, 2–10 GHz on Taconic TLY | S11 below −10 dB across all of 2–10 GHz (worst −11.2 dB at 2.5 GHz); isolation better than −25 dB; axial ratio below 1 dB; gain 4–6.4 dBi |
| Oct 2, smoke, 0.8–8 GHz on RO4003C (earlier design) | S11 below −10 dB over 98% of 0.8–8 GHz |

## Other plots

- **Return loss / VSWR per polarization**: below −10 dB (VSWR below 2) means at least 90% of the power goes into the antenna.
- **Isolation**: leakage between the two arm pairs. Below −20 dB is good.
- **Impedance / Smith chart**: sits around 200 Ω on this substrate.
- **Patterns**: two lobes (up and down) because there is no cavity yet.
- **Axial ratio**: below 3 dB means the circular mode really is circular.
- **Surface current**: in HFSS, *Field Overlays → Jsurf_…* → right-click → *Animate*. Set the legend to a log scale to see the currents on the arms.

## Next stages

1. ~~Arms + ideal feed~~
2. Balun / feed region (Klopfenstein, compact), without disturbing the pattern
3. Thin film substrate (Kapton): change the 4 `SUBSTRATE_` lines in `run_sinuous.py`
4. Cavity + absorber behind the arms (one-sided pattern)
