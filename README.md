# VLM based trajectory selector

## Setup

```sh
sudo apt-get update && sudo apt-get install -y wmctrl x11-utils gnome-screenshot
```

```sh
uv venv -p python3.10
```

```sh
source .venv/bin/activate
```

```sh
uv pip install .
```

## Set Gemini API Key

```sh
export GEMINI_API_KEY="YOUR_API_KEY"
```

## Run

```sh
python vlm_trajectory_selector.py
```
