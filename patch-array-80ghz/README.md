# 80 GHz series-fed patch array

| File | |
|---|---|
| `patch_array_original.py` | Script as shared by a labmate (was a `.txt`), kept unchanged for reference |
| `patch_array_fixed.py` | Fixed version: AEDT 2025.2, one AEDT session, far-field sphere added, gain sign fixed, safe optimizer, debug modes |

At the top of `patch_array_fixed.py`, set `MODE`:
`"geometry"` (build only) → `"single"` (one solve) → `"optimize"`.
`SMOKE_TEST = True` gives a coarse, fast mesh for debugging.

The list of changes is in the docstring at the top of the fixed file.
