# SOArm Station Training

Tools for collecting robot demonstrations, reviewing and splitting datasets, training LeRobot policies, evaluating held-out actions, and exporting versioned policy bundles. The separate SOArm Station runtime repository handles ROS control and policy execution on the robot host.

Collection uses explicit arm/leader pairs and shared camera resources. SO-101 is the current hardware integration. Training can run on a different machine from collection and deployment.

Repository: `soarm-station-training`. Companion repository: `soarm-station-runtime`.

Existing installations using `soarm-training` must reinstall this package and update commands to `soarm-station-training`.

## Requirements and installation

The bootstrap creates a Python 3.12 environment using the tested Linux CPU dependencies in `requirements.lock`, including LeRobot 0.7.0 and PyTorch 2.11.0+cpu. It requires network access and either an installed `uv` command or system Python with venv support to install `uv`.

From a clone or extracted copy of this repository, run:

```bash
scripts/bootstrap
source .venv/bin/activate
soarm-station-training --help
```

Use a suitable GPU host for substantial VLA training. The default bootstrap installs CPU PyTorch; `requirements-gpu.in` leaves the CUDA-specific PyTorch selection to the training environment. Device support and available memory must match the chosen policy and training configuration.

## Quick start: synthetic pipeline

This example requires no robot hardware. Run it in a fresh checkout or choose new dataset/output paths if the fixture already exists:

```bash
soarm-station-training record --config config/synthetic.yaml \
  --episodes 3 --seconds 1 --synthetic
soarm-station-training split local/synthetic_smoke --output local/synthetic_split.json
soarm-station-training train --config config/act_smoke.yaml --split local/synthetic_split.json
```

The fixture generates three image episodes and runs two ACT optimization steps. It checks software integration and checkpoint generation; it does not train a manipulation skill.

## Collect demonstrations

Copy the collection example and configure it for the collection host:

```bash
mkdir -p local
cp config/collection.example.yaml local/collection.yaml
```

Set a new dataset root, task description, stable device paths, logical resource IDs and calibration directories. Each follower is paired explicitly with a leader. Cameras are defined once and shared across observations. The example uses body angles in degrees and gripper position in percent.

Stop any runtime or other process owning those devices before collection. Runtime and collection use the same physical-device locks under `/tmp/soarm-station-device-locks`; use the same lock directory when customizing either application. If a service owns the devices, stop it for the collection session.

Collection connects with `calibrate=False`, so calibration must already exist. Align leaders before recording. Native LeRobot connections can configure motors; the configured disconnect policy determines whether torque is retained. The example retains follower torque on disconnect.

```bash
soarm-station-training record --config local/collection.yaml --episodes 50 --seconds 30
soarm-station-training audit datasets/sorting_session_001
lerobot-dataset-viz --repo-id local/sorting \
  --root datasets/sorting_session_001 --episode-index 0
soarm-station-training split datasets/sorting_session_001 --output local/sorting_split.json
```

Episode boundaries are operator-confirmed. Review demonstrations and task labels before training. `audit` checks named vector schemas, completed episodes, FPS and finite values. The deterministic episode-level split avoids adjacent-frame leakage; preserve the split file for reproducible experiments. Recording does not upload datasets to the Hub.

## Train and evaluate a VLA

Create a local copy of the training configuration:

```bash
cp config/smolvla.yaml local/smolvla.yaml
```

Set the dataset repo ID, output directory, batch size, step count and compute device. The configuration fine-tunes the public SmolVLA base through `lerobot-train`. `auto` selects an available CUDA, MPS or CPU device; the environment must provide a compatible PyTorch build. Trackio and Hub pushes are disabled in generated commands.

```bash
soarm-station-training train --config local/smolvla.yaml \
  --split local/sorting_split.json --dry-run
soarm-station-training train --config local/smolvla.yaml --split local/sorting_split.json
soarm-station-training train --config local/smolvla.yaml \
  --split local/sorting_split.json --resume
soarm-station-training evaluate \
  --checkpoint outputs/smolvla_sorting_v1/checkpoints/last/pretrained_model \
  --root datasets/sorting_session_001 --repo-id local/sorting \
  --split local/sorting_split.json --output local/held_out_evaluation.json --device cuda
```

The checkpoint path above matches the example output directory. Adjust it after changing `output_dir`. Offline evaluation reports held-out action MSE. Measure physical task success separately on the deployment system; action error alone does not establish manipulation success. Multi-arm collection does not imply coordinated multi-arm runtime support.

## Export a policy bundle

```bash
cp config/export.example.yaml local/export.yaml
```

Fill in the checkpoint's exact state/action feature names and order, units, camera/resource mappings, action FPS, language support and calibration revision. Checkpoint dimensions alone do not establish runtime compatibility.

```bash
soarm-station-training export \
  --checkpoint outputs/smolvla_sorting_v1/checkpoints/last/pretrained_model \
  --profile local/export.yaml --evaluation local/held_out_evaluation.json \
  --destination exports/sorting_v1
```

Export copies the complete pretrained directory, including saved processors, and hashes the artifacts in `manifest.json`. It refuses to overwrite an existing export version. Retain policy-specific tokenizer/backbone assets required for offline loading. Register the bundle as a named profile in the runtime and verify it on the target host before evaluating physical behavior.

## Repository layout and tests

| Path | Purpose |
| --- | --- |
| `src/soarm_station_training/` | Collection, dataset, training, evaluation and export implementation. |
| `config/` | Collection, synthetic, training and export examples. |
| `scripts/bootstrap` | CPU development-environment setup. |
| `tests/` | Pipeline checks. |
| `local/` | Ignored host configuration and synthetic fixtures. |
| `datasets/`, `outputs/`, `exports/` | Generated datasets, training runs and deployment bundles. |

```bash
.venv/bin/python -m pytest -q tests
```

Synthetic ACT collection, checkpoint save, held-out evaluation, export and runtime inference have been exercised. Useful SmolVLA fitting and physical task performance require a real dataset and separate evaluation; neither follows from the synthetic fixture.
