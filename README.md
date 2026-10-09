# Research-RF-Antennas

HFSS / PyAEDT scripts for RF antenna research (UNC Charlotte).

| Folder | Project |
|---|---|
| [`sinuous-uwb/`](sinuous-uwb/) | 4-arm sinuous ultra-wideband antenna, 2–10 GHz |
| [`patch-array-80ghz/`](patch-array-80ghz/) | 80 GHz series-fed patch array with optimizer (original + fixed version) |

## Opening this in PyCharm on the remote desktop

1. **Clone**: *File → New → Project from Version Control*, choose **GitHub**, sign in, pick `Research-RF-Antennas`, then **Clone**.
2. **Interpreter**: use the same Ansys Python you already use for PyAEDT. Set it under *Settings → Project → Python Interpreter*.
3. **Run**: open `sinuous-uwb/run_sinuous.py`, check `MODE` at the top, then right-click → **Run 'run_sinuous'**.
4. **Get updates**: *Git → Pull* (or the blue down-arrow in the toolbar).

Simulation outputs go to `results/` folders. These are ignored by git because HFSS files are huge. Copy the plots you need into your report.
