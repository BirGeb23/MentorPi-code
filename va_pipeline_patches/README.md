# Patches for the team's VA pipeline

These patches change files in the team's repo (`chaicenter-spot-basics/va-driven-behaviors`), not in this repo. They're kept here so they don't get lost; they should go to the team repo as a separate pull request for review. Each applies cleanly to the version of the team repo downloaded in May 2026 (checked with `git apply --check`).

| Patch | File in the team repo | Change |
|---|---|---|
| `run_offline.patch` | `va-driven-behaviors/run_offline.py` | Seek video windows by frame number instead of milliseconds (millisecond seeking failed on AVI/XVID clips, so every window got the same frames). Pass each window's own audio slice to the analyzer. |
| `test_emotions.patch` | `va-driven-behaviors/test_emotions.py` | `extract_audio_features()` gains `offset` and `duration`, so one window's audio can be read from a longer file. |
| `online_streaming_session.patch` | `va-driven-behaviors/online/streaming_session.py` | Configurable `max_workers` (default 2, unchanged behavior) and per-window queue-delay diagnostics. |

Apply them from the root of a clone of the team repo:

    FIX=~/MentorPi-code/va_pipeline_patches      # wherever this repo is cloned
    git apply $FIX/run_offline.patch
    git apply $FIX/test_emotions.patch
    git apply $FIX/online_streaming_session.patch

The MentorPi code in this repo also runs without them: `va/run_online_mentorpi.py` only uses `--max-workers` if the patched `streaming_session.py` supports it.

Two more fixes described in `docs/VA_Pipeline_Issues_Reference.md` (issue #5, the ffmpeg fallback for .mov audio, and issue #7, the CHECK_IN calm-positive split) exist only in Biruk's local working copy and are not in these patches yet.
