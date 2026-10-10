"""SkyCheck swell and gust updater.
Downloads ECMWF's free open-data wave forecast (CC BY 4.0) and saves the forecast for the
grid cells just off the NSW coast into swell.json, which the SkyCheck page reads.
Also saves ECMWF's 10 m wind gust forecast for the SkyCheck area into gusts.json.
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

def fetch(url, rng=None, tries=6):
    """One polite request at a time, with back-off. Byte-range pieces are checked: they must be exactly
    the expected size and be a GRIB message, otherwise they're fetched again (or another copy is tried)."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "skycheck-swell (iflyhigh.com.au)"})
            if rng: req.add_header("Range", f"bytes={rng[0]}-{rng[0] + rng[1] - 1}")
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if rng and (len(data) != rng[1] or not data.startswith(b"GRIB") or not data.endswith(b"7777")):
                raise ValueError(f"bad piece ({len(data)} bytes, expected {rng[1]})")
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            err = e
        except Exception as e:
            err = e
        time.sleep(min(30, 4 * (k + 1)))
    raise RuntimeError(f"Gave up on {url}: {err}")

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

def download(stream="wave", params=PARAMS, target="wave.grib2", run=None, optional=False, check=None):
    """Download only the fields we need, one small request at a time.
    stream: "wave" or "oper" (the atmosphere model). params: GRIB short names; the first one
    found in the index is used when optional=True (names differ slightly between model versions)."""
    last = None
    for ymd, hh in ([run] if run else runs_to_try()):
        for base in MIRRORS:
            root = f"{base}/{ymd}/{hh:02d}z/ifs/0p25/{stream}/{ymd}{hh:02d}0000"
            try:
                if fetch(f"{root}-{STEPS[-1]}h-{stream}-fc.index") is None:
                    print(f"{stream} {ymd} {hh:02d}z not complete on {base.split('/')[2]}", flush=True); continue
                print(f"Downloading {stream} {ymd} {hh:02d}z from {base.split('/')[2]}", flush=True)
                chunks, used = [], None
                for st in STEPS:
                    if optional and st == 0: continue          # gusts are a max over the previous hours: none at step 0
                    idx = fetch(f"{root}-{st}h-{stream}-fc.index")
                    if idx is None: raise RuntimeError(f"missing index for step {st}")
                    entries = [json.loads(l) for l in idx.decode().splitlines() if l.strip()]
                    wanted = params if not optional else ([used] if used else params)
                    for p in wanted:
                        e = next((x for x in entries if x.get("param") == p), None)
                        if e is None:
                            if optional: continue
                            raise RuntimeError(f"{p} missing at step {st}")
                        chunks.append(fetch(f"{root}-{st}h-{stream}-fc.grib2", (int(e["_offset"]), int(e["_length"]))))
                        if optional: used = p; break
                    time.sleep(0.05)
                if optional and not used:
                    print(f"  none of {params} published in {stream}", flush=True); return None, None
                with open(target, "wb") as fh:
                    for c in chunks: fh.write(c)
                print(f"Downloaded {len(chunks)} {stream} fields", flush=True)
                info = (ymd, hh, used)
                if check: check(target, info)      # raises if the file can't be read; then the next copy is tried
                return target, info
            except Exception as ex:
                print(f"  failed on {base.split('/')[2]}: {str(ex)[:160]}", flush=True)
                last = ex
    if optional: return None, None
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

GUST_BOX = dict(lat=(-35.0, -31.0), lon=(149.75, 153.0))   # Sydney to the Upper Hunter and Port Stephens
GUST_PARAMS = ["10fg", "fg310", "i10fg"]                       # 10 m wind gust (name varies by model version)

def gust_grid(ds, run_ms):
    var = list(ds.data_vars)[0]
    sub = ds[var].sel(latitude=slice(GUST_BOX["lat"][1], GUST_BOX["lat"][0]), longitude=slice(*GUST_BOX["lon"]))
    steps_h = (sub["step"].values / np.timedelta64(1, "h")).astype(int)
    lats, lons = sub["latitude"].values, sub["longitude"].values
    v = sub.values  # (step, lat, lon) in m/s
    return {
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "run": run_ms, "source": "ECMWF open data, 10 m wind gust (CC BY 4.0)",
        "note": "g[i] is the strongest gust (m/s x 10) in the hours up to times[i]",
        "times": [int(run_ms + h * 3600_000) for h in steps_h],
        "lat0": round(float(lats[0]), 2), "lon0": round(float(lons[0]), 2), "dlat": round(float(lats[1] - lats[0]), 2), "dlon": round(float(lons[1] - lons[0]), 2),
        "nlat": int(len(lats)), "nlon": int(len(lons)),
        "g": [[[None if np.isnan(x) else int(round(float(x) * 10)) for x in row] for row in v[k]] for k in range(len(steps_h))],
    }

def open_param(path, short):
    """Read one parameter. GRIB decoders sometimes name a field slightly differently from the download
    index (e.g. gusts as 10fg, 10fg3 or 10fg6 depending on the time window), so fall back to matching
    by prefix and joining the pieces along the forecast steps."""
    import xarray as xr, cfgrib
    key = short.lstrip("0123456789i")[:2] if short[0].isdigit() or short.startswith("i") else short
    parts = []
    for d in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
        for v in d.data_vars:
            sn = str(d[v].attrs.get("GRIB_shortName", v))
            if sn == short or sn.startswith(short) or (key and key in sn):
                da = d[v]
                if "step" not in da.dims: da = da.expand_dims("step")
                parts.append(da.rename(short))
    if not parts: raise ValueError(f"{short} not found in {path}")
    da = xr.concat(parts, "step").sortby("step") if len(parts) > 1 else parts[0]
    _, keep = np.unique(da["step"].values, return_index=True)
    return da.isel(step=np.sort(keep))

def read_wave(path, info=None):
    import xarray as xr
    ds = xr.merge([open_param(path, p) for p in PARAMS], compat="override", join="override")
    if ds["swh"].sizes.get("step", 0) < 20: raise ValueError("too few forecast steps in wave file")
    return ds

if __name__ == "__main__":
    import xarray as xr
    f, _ = download(check=lambda p, i: read_wave(p))
    ds = read_wave(f)
    out = build(ds)
    if len(out["cells"]) < 10:
        raise SystemExit("Too few coastal cells found, not overwriting swell.json")
    with open("swell.json", "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"Saved {len(out['cells'])} coastal cells, {len(out['times'])} times, run {out['run']}")

    # gusts: same model run as the waves, so the two line up
    run_dt = dt.datetime.fromtimestamp(out["run"] / 1000, dt.timezone.utc)
    try:
        gf, info = download("oper", GUST_PARAMS, "gust.grib2", run=(run_dt.strftime("%Y%m%d"), run_dt.hour), optional=True,
                            check=lambda p, i: open_param(p, i[2]))
        if gf:
            gds = open_param(gf, info[2]).to_dataset()
            gout = gust_grid(gds, out["run"])
            with open("gusts.json", "w") as fh:
                json.dump(gout, fh, separators=(",", ":"))
            print(f"Saved gusts ({info[2]}): {gout['nlat']}x{gout['nlon']} grid, {len(gout['times'])} times")
    except Exception as ex:
        print(f"Gusts skipped this run: {str(ex)[:200]}")   # swell.json is still saved
