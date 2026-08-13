# Fixed-camera demo pipeline

This set contains three candidate traffic-light cycles from one fixed camera.
Each candidate has stopped-queue frames in `before/` and ordered post-prompt
frames in `after/`. The last `before` image is repeated as the first `after`
image and is automatically de-duplicated by the builder.

## Roboflow setup

The public [Red Light Racer Vehicle Detection](https://app.roboflow.com/mani_samva-yahoo-co-in/red-light-racer-vehicle-detection)
Project Deployment was provisioned through the Roboflow MCP. It uses SAM3 with
these classes:

- `car`
- `truck`
- `bus`
- `motorcycle` (optional)

The hosted image endpoint is stateless. Do not add ByteTrack or a line counter;
car association and finish-line crossing are handled locally by this repo.

Set credentials in PowerShell without committing them:

```powershell
$env:ROBOFLOW_API_KEY = "..."
$env:ROBOFLOW_WORKSPACE = "mani_samva-yahoo-co-in"
$env:ROBOFLOW_WORKFLOW_ID = "red-light-racer-vehicle-detection"
```

Build the cleanest sample first:

```powershell
.\.venv\Scripts\python.exe tools\build_fixed_races.py demo_set_1 --candidate cand_002
```

The default command writes preview artifacts under
`data/generated/demo_set_1/`. API results are cached by image hash, so geometry
and matching can be tuned without paying for repeated inference.

After reviewing the generated round:

```powershell
.\.venv\Scripts\python.exe tools\build_fixed_races.py demo_set_1 --candidate cand_002 --detections data\generated\demo_set_1\detections.json --promote --round-set roboflow
```

`--promote` writes to `data/round_sets/<name>/`; it never replaces the
classic rounds in `data/rounds.json`.

## Switching round sets

Classic remains the default:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app
```

Start the app with the Roboflow-generated set:

```powershell
$env:ROUND_SET = "roboflow"
.\.venv\Scripts\python.exe -m uvicorn app.main:app
```

Restart the server after changing `ROUND_SET`. Set it back to `classic` (or
unset it) to return to the original rounds.
