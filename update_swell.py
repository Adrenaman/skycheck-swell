"""SkyCheck swell updater.
Downloads ECMWF's free open-data wave forecast (CC BY 4.0) and saves the forecast for the
grid cells just off the NSW coast into swell.json, which the SkyCheck page reads.
Runs on GitHub Actions twice a day (see .github/workflows/swell.yml)."""
import json, math, sys, datetime as dt
import numpy as np

BOX = dict(lat=(-37.6, -28.1), lon=(149.8, 154.2))   # NSW coast and offshore
STEPS = list(range(0, 145, 3)) + list(range(150, 175, 6))  # 0-144 h every 3 h, then 6-hourly to 7 days
PARAMS = ["swh", "pp1d", "mwd"]   # wave height (m), peak period (s), mean direction (deg, coming from)

def download(target="wave.grib2"):
    from ecmwf.opendata import Client
    client = Client(source="ecmwf")
    last = None
    for run in (12, 0):   # the 00 and 12 UTC runs go out to 10 days; use the newest one available
        try:
            client.retrieve(time=run, stream="wave", type="fc", step=STEPS, param=PARAMS, target=target)
            return target
        except Exception as e:
            last = e
    raise SystemExit(f"Download failed: {last}")

def coastal_cells(ds):
    """For each grid row, keep the first sea cell east of the coast (the wave model has no data on land)."""
    sub = ds.sel(latitude=slice(BOX["lat"][1], BOX["lat"][0]), longitude=slice(*BOX["lon"]))
    swh = sub["swh"]
    first = swh.isel(step=0).values
    lats, lons = sub["latitude"].values, sub["longitude"].values
    cells = []
    for i, la in enumerate(lats):
        row = first[i]
        sea = np.where(~np.isnan(row))[0]
        if not len(sea):
            continue
        j = sea[0]
        if j == 0:          # open ocean right to the western edge: no coast in this row
            continue
        def series(name, scale, nd=0):
            v = sub[name].values[:, i, j]
            return [None if np.isnan(x) else round(float(x) * scale, nd) if nd else int(round(float(x) * scale)) for x in v]
        cells.append({
            "lat": round(float(la), 2), "lng": round(float(lons[j]), 2),
            "h": series("swh", 10),          # decimetres (1.4 m -> 14)
            "p": series("pp1d", 1),          # seconds
            "d": series("mwd", 1),           # degrees, direction the waves come from
        })
    return cells

def build(ds):
    base = np.datetime64(ds["time"].values).astype("datetime64[ms]").astype("int64")
    steps_h = (ds["step"].values / np.timedelta64(1, "h")).astype(int)
    return {
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "run": int(base),
        "source": "ECMWF open data wave forecast (CC BY 4.0)",
        "times": [int(base + h * 3600_000) for h in steps_h],
        "cells": coastal_cells(ds),
    }

if __name__ == "__main__":
    import xarray as xr
    f = download()
    ds = xr.open_dataset(f, engine="cfgrib")
    out = build(ds)
    if len(out["cells"]) < 10:
        raise SystemExit("Too few coastal cells found, not overwriting swell.json")
    with open("swell.json", "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"Saved {len(out['cells'])} coastal cells, {len(out['times'])} times, run {out['run']}")
