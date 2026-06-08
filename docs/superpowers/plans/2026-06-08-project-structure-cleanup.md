# Project Structure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the robot curl repository so active code, scripts, tests, assets, docs, and legacy experiments are easy to find without changing controller behavior.

**Architecture:** Move active reusable code into `robot_curl/`, command-line entrypoints into `scripts/`, tests into `tests/`, MJCF/XML assets into `assets/`, design notes into `docs/`, and old exploratory scripts into `legacy/`. Keep output directories (`quick_runs/`, `ppo_logs/`, `ppo_models/`) ignored and unmoved.

**Tech Stack:** Python, MuJoCo, Gymnasium, Stable-Baselines3, NumPy, PowerShell.

---

### Task 1: Move Files Into Clear Directories

**Files:**
- Create: `robot_curl/__init__.py`
- Move: `test_mujoco/env.py` to `robot_curl/env.py`
- Move: `test_mujoco/policy_search.py` to `robot_curl/policy_search.py`
- Move: active entrypoints from `test_mujoco/` to `scripts/`
- Move: active tests from `test_mujoco/` to `tests/`
- Move: XML assets from `test_mujoco/` to `assets/`
- Move: historical experiments from `test_mujoco/` to `legacy/`
- Move: design notes from `test_mujoco/` to `docs/`

- [x] **Step 1: Create package and destination directories**

Run: `New-Item -ItemType Directory -Force robot_curl,scripts,tests,assets,docs,legacy`

- [x] **Step 2: Move files with git**

Use `git mv` for tracked files so history is preserved.

### Task 2: Fix Imports And Asset Paths

**Files:**
- Modify: `robot_curl/env.py`
- Modify: `robot_curl/policy_search.py`
- Modify: `scripts/*.py`
- Modify: `tests/*.py`

- [x] **Step 1: Replace local imports**

Use `from robot_curl.env import QuadrupedFoldEnv` and `from robot_curl.policy_search import ...`.

- [x] **Step 2: Resolve XML path relative to package root**

In `robot_curl/env.py`, load `assets/quadruped.xml` from the repository root instead of assuming the process starts inside `test_mujoco/`.

### Task 3: Update User-Facing Commands

**Files:**
- Modify: `README.md`

- [x] **Step 1: Replace `test_mujoco\...` commands with `scripts\...` and `tests\...` paths**

Document quick train, model evaluation, CEM baseline, and feedback CEM commands.

### Task 4: Verify Behavior

**Files:**
- No source edits unless verification finds path/import issues.

- [x] **Step 1: Compile active Python modules**

Run: `python -m py_compile robot_curl\env.py robot_curl\policy_search.py scripts\quick_train.py scripts\evaluate_curl.py scripts\cem_optimize.py`

- [x] **Step 2: Run lightweight tests**

Run the manual test loader over `tests/test_curl_reward.py`, `tests/test_policy_search.py`, and `tests/test_cem_optimize.py`.

- [x] **Step 3: Run a representative baseline evaluation**

Run: `python scripts\evaluate_curl.py --policy cem --episodes 3`
