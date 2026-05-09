# Gesture-Controlled Presentation Assistant

This project uses a camera, MediaPipe pose/hand tracking, a hand-built PCA + weighted kNN classifier, and keyboard shortcuts to control a presentation with gestures.

It currently supports 4 actions:

- `right_arm` -> next slide
- `left_arm` -> previous slide
- `five_fingers` -> Canva countdown shortcut
- `thumb_up` -> Canva confetti shortcut

The repo also includes:

- a simple OpenCV CLI runner
- a Tkinter GUI
- a dataset collection flow
- a training script
- a pipeline visualizer for demos / reports

## Requirements

- Python 3.11 or 3.12 recommended
- A webcam or DroidCam
- A desktop session that can open GUI windows
- Canva in Present mode if you want live shortcut control

Install dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use `python3.13`, some CV packages may lag behind wheel support. This repo should be set up with `python3.12` on this machine.

If you already created `venv` with a different Python version, delete it and recreate it cleanly instead of reusing a mixed environment.

## Quick Start

If you just want to try the current model:

```bash
python3 scripts/setup_mediapipe_models.py
python3 main.py
```

If the preview is black or the wrong camera opens, try another source:

```bash
python3 main.py --camera 0
python3 main.py --camera 1
```

Or launch the GUI:

```bash
python3 app.py
```

Press `ESC` in the OpenCV windows to quit.

On newer MediaPipe installs, the first run may download the official `.task` model bundles into `models/mediapipe/`. You can do that explicitly ahead of time with `python3 scripts/setup_mediapipe_models.py`.

## Full Workflow

### 1. Choose the right camera

List available cameras:

```bash
python3 scripts/list_cameras.py
```

Then update `CAMERA_SOURCE` in [config.py](/Users/ben/Documents/GitHub/gesture_presenter_LA/config.py:1) if needed.

### 1.5. Prepare MediaPipe model bundles

If your installed `mediapipe` package uses the newer Tasks API, this repo needs local `.task` files for pose and hand tracking. The helper script downloads them into `models/mediapipe/`:

```bash
python3 scripts/setup_mediapipe_models.py
```

The app can also download them automatically on first run when `MEDIAPIPE_AUTO_DOWNLOAD_MODELS = True`.

### 2. Collect training data

Collect gesture sessions for one person:

```bash
python3 -m data.collect_data
```

What this does:

- prompts for a person name
- records one session per gesture
- saves feature vectors, not raw video
- stores new recordings as session files under `data/dataset/<person>/<gesture>/`

The collector now filters out obviously bad samples:

- no pose detected
- low pose visibility
- missing hand data for hand-required gestures

### 3. Verify the dataset

```bash
python3 -m data.verify_dataset
```

This prints sample counts and basic feature statistics for each gesture.

### 4. Train the model

```bash
python3 -m models.train_pca
```

What the trainer now does:

- loads all available gesture sessions
- prefers person-level holdout if multiple people exist
- otherwise prefers session-level holdout
- only falls back to contiguous within-session holdout when the dataset is too small
- selects `n_components` on validation data, not on the test set
- reports an honest test accuracy
- then saves a deployment model trained on all available data

The saved model is written to:

- `models/pca_model.npz`

### 5. Run the app

CLI preview:

```bash
python3 main.py
```

Tkinter GUI:

```bash
python3 app.py
```

Pipeline visualizer:

```bash
python3 -m visualization.pipeline_visualizer
```

## Runtime Behavior

The runtime path is:

1. capture camera frame
2. resize for inference
3. detect pose + hands with MediaPipe
4. extract a 13-D feature vector
5. standardize features
6. project into PCA space
7. classify with weighted kNN
8. smooth predictions across frames
9. trigger a key press if the gesture is stable and allowed

Important safeguards added in this refactor:

- frame-to-frame gesture smoothing
- release-to-rearm trigger logic
- cooldown between actions
- optional focus guard for Canva
- separate tracking states like `no_pose` and `low_visibility`

## Configuration

Most behavior lives in [config.py](/Users/ben/Documents/GitHub/gesture_presenter_LA/config.py:1).

The most important settings are:

- `CAMERA_SOURCE`
- `CAMERA_IS_MIRRORED`
- `MEDIAPIPE_BACKEND`
- `MEDIAPIPE_AUTO_DOWNLOAD_MODELS`
- `CONFIDENCE_THRESHOLD`
- `TRIGGER_STABLE_FRAMES`
- `TRIGGER_COOLDOWN`
- `TARGET_APP_NAME`
- `ENFORCE_TARGET_APP_FOCUS`
- `PCA_N_COMPONENTS`
- `KNN_K`

## Current Limitations

- The checked-in dataset is still small and appears to be from one person.
- Honest evaluation is implemented, but cross-person generalization cannot be proven until you collect data from more people.
- Focus guarding is strongest on macOS because it uses `osascript` to detect the frontmost app. Other platforms currently fall back to a softer behavior.
- The feature extractor is still intentionally compact and simple because this is a teaching/demo project, not a production gesture system.

## Repo Layout

```text
.
├── app.py
├── main.py
├── config.py
├── data/
│   ├── collect_data.py
│   ├── verify_dataset.py
│   └── dataset/
├── models/
│   ├── train_pca.py
│   └── pca_model.npz
├── scripts/
│   └── list_cameras.py
├── src/
│   ├── action_trigger.py
│   ├── camera.py
│   ├── dataset.py
│   ├── detector.py
│   ├── display.py
│   ├── feature_extractor.py
│   ├── gesture_stabilizer.py
│   ├── pca_classifier.py
│   ├── pipeline.py
│   └── preprocessor.py
├── tests/
└── visualization/
    └── pipeline_visualizer.py
```

## Tests

Run the lightweight automated tests with:

```bash
python3 -m unittest discover -s tests -v
```

These cover:

- PCA fit / save / load
- trigger rearm / focus guard behavior
- dataset splitting strategy

## Troubleshooting

If the camera does not open:

- run `python3 scripts/list_cameras.py`
- set the correct `CAMERA_SOURCE`
- make sure no other app is locking the camera

If gestures are recognized but Canva does not change slides:

- make sure Canva is actually focused
- check `TARGET_APP_NAME` and `ENFORCE_TARGET_APP_FOCUS`
- try the CLI runner first because it is the simplest path

If left/right gestures seem swapped:

- toggle `CAMERA_IS_MIRRORED` in `config.py`

If training still looks unrealistically perfect:

- that is likely a dataset issue, not a code issue
- collect multiple sessions per gesture
- collect data from multiple people
- avoid recording every session in exactly the same pose, camera angle, and lighting
