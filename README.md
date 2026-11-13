# SOARM LeRobot training

Separate demonstration, dataset, experiment and training repository. Production ROS control lives in `../ros_system`; this repository does not add dataset buttons or a training-job service to production. No Git remote is configured and nothing has been pushed. Initial implementation was produced with Codex assistance.

The Pi has a dedicated Python 3.12 `.venv`. `requirements.lock` freezes the tested Linux CPU environment, including LeRobot 0.7.0 / PyTorch 2.11.0+cpu. For substantial SmolVLA fitting, run this same repository/dataset on a suitable GPU Linux host; the Pi CPU smoke test does not establish useful VLA training throughput. `requirements-gpu.in` deliberately leaves the CUDA-specific PyTorch installation to the training host.

```bash
scripts/bootstrap
source .venv/bin/activate
soarm-training --help
```

## Collection and review

Stop the production service and disable its automatic restart for the shared-hardware collection session. Use the same `/tmp/soarm-device-locks` convention on both repositories. The collector acquires every physical lock before opening devices. Each configured follower is paired explicitly with a leader; cameras are configured once outside the arms. Stock LeRobot robot/teleoperator configurations provide the first SO-101 adapter and extension points.

Copy `config/collection.example.yaml` to ignored `local/collection.yaml`; configure actual ports, logical IDs, calibration directories, native units, task and a **new** dataset root. The example uses body degrees and separate gripper percent. Calibrate hardware separately; collection calls `connect(calibrate=False)` and preserves the configured native disconnect policy. Align leaders before recording. The native follower may configure motors when collection connects, so this is a supervised collection workflow, not read-only monitoring.

```bash
soarm-training record --config local/collection.yaml --episodes 50 --seconds 30
soarm-training audit datasets/sorting_session_001
lerobot-dataset-viz --repo-id local/sorting --root datasets/sorting_session_001 --episode-index 0
soarm-training split datasets/sorting_session_001 --output local/sorting_split.json
```

Episode boundaries are operator-confirmed. Source observations and sent actions are different fields; sent actions are not proof of applied movement or task success. Recording never pushes to a Hub. Review demonstrations and retain only supportable task labels. `audit` checks named vector schemas, completed episodes, FPS and finite values; a deterministic episode-level split avoids adjacent-frame leakage. Preserve the split file when rerunning experiments.

## VLA fitting and evaluation

`config/smolvla.yaml` fine-tunes the public SmolVLA base through stock `lerobot-train`. Set the dataset repo ID, output directory, batch size/steps and compute device for the selected host. `auto` chooses available CUDA/MPS/CPU. Trackio and Hub pushes are disabled in generated commands.

```bash
soarm-training train --config config/smolvla.yaml --split local/sorting_split.json --dry-run
soarm-training train --config config/smolvla.yaml --split local/sorting_split.json
soarm-training train --config config/smolvla.yaml --split local/sorting_split.json --resume
soarm-training evaluate --checkpoint outputs/smolvla_sorting_v1/checkpoints/last/pretrained_model \
  --root datasets/sorting_session_001 --repo-id local/sorting --split local/sorting_split.json \
  --output local/held_out_evaluation.json --device cuda
```

Offline evaluation reports held-out action MSE and explicitly labels physical task outcome untested. Evaluate manipulation success separately in production using the correct resource/profile mappings. The deployment currently executes independent single-arm profiles; collecting multiple arms does not automatically enable coordinated bimanual deployment.

## Immutable production export

Fill `config/export.example.yaml` with the checkpoint's exact feature names/order, units, camera/resource mapping, trained action FPS, language support and calibration revision. Checkpoint dimensions alone do not establish compatibility.

```bash
soarm-training export --checkpoint outputs/smolvla_sorting_v1/checkpoints/last/pretrained_model \
  --profile local/export.yaml --evaluation local/held_out_evaluation.json \
  --destination exports/sorting_v1
```

Exports refuse to overwrite an existing version. They copy the complete pretrained directory, including saved processor configuration/state, then hash every artifact into `manifest.json`. Production verifies hashes and profile mappings before use. Preserve any policy-specific tokenizer/backbone assets needed for offline loading; verify the chosen VLA bundle on the destination host before physical deployment. A training dataset or ROS bag is not an executable policy bundle.

## Reproduced integration fixture

```bash
soarm-training record --config config/synthetic.yaml --episodes 3 --seconds 1 --synthetic
soarm-training split local/synthetic_smoke --output local/synthetic_split.json
soarm-training train --config config/act_smoke.yaml --split local/synthetic_split.json
```

This fixture completed on the Pi: three five-frame image episodes, two ACT optimization steps, checkpoint save, held-out action evaluation, immutable export, local runtime inference and native Zenoh inference. It demonstrates software integration, not a trained manipulation skill. SmolVLA command generation is checked; a useful VLA has not been fitted because no real task dataset was collected in this installation.
