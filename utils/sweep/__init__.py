"""Minimal, self-contained subset of the Sharpe-lab ``measureme.sweep`` package,
vendored so this figure repository has no dependency on the lab acquisition
stack. Only the read-side helpers are included:

    from sweep import sweep_load as sl   # sl.pload / sl.load_meta / sl.load
    from sweep import raster             # raster.pcolorize_data

Both modules depend only on numpy + the standard library. They read the sweep
directory format (``<id>/metadata.json`` + ``<id>/data.tsv[.gz]``) deposited in
data/raw_sweeps/.
"""
