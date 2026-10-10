"""SkyCheck swell updater.
Downloads ECMWF's free open-data wave forecast (CC BY 4.0) and saves the forecast for the
grid cells just off the NSW coast into swell.json, which the SkyCheck page reads.
Runs on GitHub Actions twice a day (see .github/workflows/swell.yml)."""
import json, math, sys, datetime as dt
import numpy as np

BOX = dict(lat=(-37.6, -28.1), lon=(149.8, 154.2))   # NSW coast and offshore
STEPS = list(range(0, 145, 3)) + list(range(150, 175, 6))  # 0-144 h every 3 h, then 6-hourly to 7 days
PARAMS = ["swh", "pp1d", "mwd"]   # wave height (m), peak period (s), mean direction (deg, coming from)

import time, urllib.request, urllib.error

# ECMWF open data and its official cloud copies (same files, same layout)
MIRRORS = [
    "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
    "https://storage.googleapis.com/ecmwf-open-data",
    "https://data.ecmwf.int/forecasts",
]

def fetch(url, rng=None, tries=4):
    """One polite request at a time, short back-off, then give up so another copy can be tried."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "skycheck-swell (iflyhigh.com.au)"})
            if rng: req.add_header("Range", f"bytes={rng[0]}-{rng[0] + rng[1] - 1}")
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            wait = 3 * (k + 1)
        except Exception:
            wait = 3 * (k + 1)
        time.sleep(wait)
    raise RuntimeError(f"Gave up on {url}")

def runs_to_try():
    """Newest 00 or 12 UTC runs first (those go out 10 days)."""
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for back in range(0, 3):
        d = (now - dt.timedelta(days=back)).date()
        for hh in (12, 0):
            if dt.datetime(d.year, d.month, d.day, hh, tzinfo=dt.timezone.utc) + dt.timedelta(hours=7) < now:
                out.append((d.strftime("%Y%m%d"), hh))
    return out

def download(target="wave.grib2"):
    last = None
    for ymd, hh in runs_to_try():
        for base in MIRRORS:
            root = f"{base}/{ymd}/{hh:02d}z/ifs/0p25/wave/{ymd}{hh:02d}0000"
            try:
                # make sure the run is complete before downloading it
                if fetch(f"{root}-{STEPS[-1]}h-wave-fc.index") is None:
                    print(f"{ymd} {hh:02d}z not complete on {base.split('/')[2]}", flush=True); continue
                print(f"Downloading {ymd} {hh:02d}z from {base.split('/')[2]}", flush=True)
                chunks = []
                for st in STEPS:
                    idx = fetch(f"{root}-{st}h-wave-fc.index")
                    if idx is None: raise RuntimeError(f"missing index for step {st}")
                    entries = [json.loads(l) for l in idx.decode().splitlines() if l.strip()]
                    for p in PARAMS:
                        e = next((x for x in entries if x.get("param") == p), None)
                        if e is None: raise RuntimeError(f"{p} missing at step {st}")
                        chunks.append(fetch(f"{root}-{st}h-wave-fc.grib2", (int(e["_offset"]), int(e["_length"]))))
                    time.sleep(0.05)
                with open(target, "wb") as fh:
                    for c in chunks: fh.write(c)
                print(f"Downloaded {len(chunks)} fields", flush=True)
                return target
            except Exception as ex:
                print(f"  failed on {base.split('/')[2]}: {str(ex)[:160]}", flush=True)
                last = ex
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
    ds = xr.merge([xr.open_dataset(f, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": p}, "indexpath": ""})[[p]]
                   for p in PARAMS], compat="override")
    out = build(ds)
    if len(out["cells"]) < 10:
        raise SystemExit("Too few coastal cells found, not overwriting swell.json")
    with open("swell.json", "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"Saved {len(out['cells'])} coastal cells, {len(out['times'])} times, run {out['run']}")
