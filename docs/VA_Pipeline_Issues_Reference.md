# VA Pipeline — Code Issues Reference

Prepared for team discussion. Line numbers refer to Biruk's patched working copy of `va-driven-behaviors`, so they can differ from the version on GitHub.

**Status note (October 2026):** the fixes described under "Issues already fixed" (#5, #6, #7) were made in a local working copy and are **not yet in the GitHub repo**. They will come in a separate pull request for review. The MentorPi code in this repo works without them.

---

## Open issues needing team input

### 1. Model checkpoint only loads 77/85 parameters — architecture mismatch

**File:** `test_emotions.py`, function `load_model()`, lines 576–605 (diagnostic added at lines 581–603)

```python
576:    state_dict = checkpoint['net']
577:    model_state = model.state_dict()
578:    loaded_count = 0
579:
580:    missing_keys = []
581:    for key in model_state.keys():
582:        if key in state_dict and model_state[key].shape == state_dict[key].shape:
583:            model_state[key] = state_dict[key]
584:            loaded_count += 1
585:        else:
586:            reason = "not in checkpoint" if key not in state_dict else (
587:                f"shape mismatch: model={tuple(model_state[key].shape)} "
588:                f"vs checkpoint={tuple(state_dict[key].shape)}"
589:            )
590:            missing_keys.append((key, reason))
591:
592:    model.load_state_dict(model_state, strict=False)
593:    print(f"  Loaded {loaded_count}/{len(model_state)} parameters")
```

**What's happening:** `model = JointCAM(...)` is constructed with fresh random weights, then this loop overwrites only the parameters whose name *and* shape match the checkpoint. 8 of 85 parameter tensors never match, so they keep their random initial values. Since no `torch.manual_seed()` is set anywhere in the codebase, those 8 tensors get **different random values every process run**.

**Root cause (confirmed via diagnostic output on `jointcam_finetuned_v4.pt`):**

| Missing parameter | Model shape | Checkpoint shape | Meaning |
|---|---|---|---|
| `video_extract.init_hidden` | (4, 256) | (8, 256) | Not a layer-count mismatch: the checkpoint's LSTMs also have 2 layers (`_l0`, `_l1` weights only). The 8 rows most likely store the initial h0 and c0 together, while the current code stores only h0 (4 rows). Changing `num_layers` to 4 would break 16 more weights. |
| `audio_extract.init_hidden` | (4, 256) | (8, 256) | Same h0/c0 storage difference, audio branch |
| `Joint.init_hidden` | (4, 256) | (8, 256) | Same h0/c0 storage difference, fusion branch |
| `video_extract.linear.weight` | (512, 512) | (512, 1024) | Model expects 512-dim video features in; checkpoint trained on 1024-dim. The current code averages ImageNet ResNet-50 features (2048) down to 512, so the visual features also differ from the ones the model was trained on. |
| `Joint.rnn.weight_ih_l0` / `_reverse` | (1024, 512) | (1024, 1024) | Same 512-vs-1024 input-dimension mismatch, fusion RNN |
| `video_attn.attn.affine_out.weight` / `.bias` | — | not in checkpoint | Current attention module has no equivalent in the checkpoint at all |

Checkpoint also has 5 unused keys with no home in the current model: `video_attn.attn.affine_feat.{weight,bias}`, `affine_g.weight`, `affine_h.weight`, `affine_v.weight` — a completely different attention design than what's currently implemented (single `affine_out`). Their shapes (49 = 7×7) suggest audio-guided attention over a 7×7 grid of image regions. The co-attention weights do load, but its `forward()` is a pass-through and its outputs are not used in the joint fusion.

**Likely fix:** recover the original model class and feature-extraction code (the `models/README.md` points to `Spot_Mini_Simulation/spot_mini_mini/spot_bullet/src/emotion/`) rather than reshaping layers. `jointcam_model.pt` stores validation scores (`best_Val_accV` 0.715, `best_Val_accA` 0.688), so the original setup did perform well.

**Confirmed NOT a wrong-checkpoint issue:** all three `.pt` files in the repo (`va-driven-behaviors/models/jointcam_finetuned_v4.pt`, `emotion-poc/jointcam_model.pt`, `va-testing/Model/jointcam_model.pt`) score identically 77/85 against the current code. They agree with each other; none agree with the current `JointCAM`/`BiLSTMExtractor` class definitions.

**Relevant current code** (`test_emotions.py`, unaffected by any patch, original location):
```python
31:class BiLSTMExtractor(nn.Module):
32:    """Bidirectional LSTM for temporal feature extraction."""
33:
34:    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
```
`class JointCAM` begins at line 170 — that's where `video_extract`, `audio_extract`, `Joint`, and `video_attn` get instantiated using this class; worth checking there for exactly how/whether `num_layers` and feature dimensions are wired through.

**Confirmed impact:** affects **both** offline (`run_offline.py`) and online (`run_online.py`) — both call the same `load_model()`.

**Questions for the team:** Was the architecture (LSTM depth, feature dims, attention design) intentionally changed after `jointcam_finetuned_v4.pt` was trained? Is there a newer, matching checkpoint that should be used instead? Does anyone have the training-time version of this code, or documentation of the original architecture?

---

### 2. `compute_trends()` hardcodes confidence to 0.5 for "mixed" trend, ignoring volatility

**File:** `pipeline/emotion_analyzer.py`, lines 630–654

```python
630:    # Determine direction and confidence
631:    if slope_sign == delta_sign and slope_significant and delta_significant:
632:        # Agreement: clear trend
633:        if slope_sign > 0:
634:            direction = 'rising'
635:        elif slope_sign < 0:
636:            direction = 'falling'
637:        else:
638:            direction = 'stable'
639:        confidence = 0.8 + 0.2 * (1.0 - min(volatility / (volatility_threshold * 2), 1.0))
640:    elif slope_sign != delta_sign or (not slope_significant and not delta_significant):
641:        # Disagreement or both insignificant
642:        if high_volatility:
643:            direction = 'uncertain'
644:            confidence = 0.3
645:        else:
646:            direction = 'mixed'
647:            confidence = 0.5
648:    else:
649:        # One is significant, other is not
650:        if slope_significant:
651:            direction = 'rising' if slope_sign > 0 else 'falling'
652:        else:
653:            direction = 'rising' if delta_sign > 0 else 'falling'
654:        confidence = 0.6
```

**What's happening:** line 639 scales confidence with actual volatility for `rising`/`falling` — tighter data (lower volatility) legitimately earns higher confidence, up to 1.0. But line 647, the `mixed` branch, is a flat constant `0.5` regardless of how stable the data actually is. In our test runs, volatility was as low as 0.002–0.015 (very stable) and confidence was still capped at exactly 0.500 every time — the same value it'd get if volatility were much higher (just under the `high_volatility` cutoff). This is inconsistent with how the `rising`/`falling` branch already handles the same situation.

**Suggested direction:** apply the same kind of volatility-aware scaling used on line 639 to the `mixed` branch, so low-volatility "mixed" (genuinely flat/stable) reads as higher confidence than volatility close to the high-volatility threshold.

---

### 3. `compute_state_label()` collapses neutral arousal into "-low-arousal"

**File:** `pipeline/emotion_analyzer.py`, lines 679–746 (relevant branch at 736–739)

```python
736:    elif a_cat == 'neutral-arousal':
737:        # To match the mapper's expected 8-state model,
738:        # append 'low-arousal' to significant valence
739:        state_label = f"{v_cat}-low-arousal"
```

**What's happening:** when arousal is genuinely in the neutral band (between calibration P30 and P70) but valence is significant (positive or negative), this collapses it into the same label as truly low arousal. The comment says this is deliberate — done to match `spot_reaction_mapper.py`'s substring-matching dispatcher, which only recognizes `high-arousal`/`low-arousal`/exact `neutral`, with no branch for "significant valence + neutral arousal."

**Current workaround (already applied):** rather than changing this label logic (which could break the mapper's dispatch), we differentiated behavior downstream in `spot_reaction_mapper.py` using the raw arousal float value instead of the label string — see fix #4 below. The label itself still collapses arousal categories; this only means `CHECK_IN`'s *intensity* now varies correctly. If any other part of the codebase reads this label and assumes it means truly low arousal, that assumption is still wrong.

**Question for the team:** should the label itself eventually be made honest (e.g. a real `positive-neutral-arousal` state), with `spot_reaction_mapper.py`'s dispatcher expanded to handle it explicitly? That's a bigger, more thorough fix than the parameter-tuning workaround we did.

---

### 4. Two parallel, inconsistent implementations of the 5-intent policy

**File:** `pipeline/intent_selector.py` (lines 27, 64) vs. `pipeline/spot_reaction_mapper.py` (embedded inside `map_to_action()`)

```python
# pipeline/intent_selector.py
27:class IntentSelector:
64:    def select_intent(self,
```

```python
# online/window_analyzer.py — IntentSelector is constructed...
48:        self.intent_selector  = IntentSelector(

# ...but only reaction_mapper is ever actually called:
178:            reaction_action, explanation = self.reaction_mapper.map_to_action(
```

**What's happening:** `IntentSelector.select_intent()` is never called anywhere in the codebase — confirmed by searching for `select_intent(` across the whole repo. `window_analyzer.py` constructs an `IntentSelector` instance (line 48) but the actual intent that ends up driving robot behavior comes entirely from a second, separately-maintained copy of the same branching logic inside `spot_reaction_mapper.py`'s `map_to_action()`. This means a future change to one won't automatically apply to the other, and anyone reading `intent_selector.py` expecting it to be "the" intent logic will be looking at dead code.

**Suggested direction:** either wire `IntentSelector` into `window_analyzer.py` and remove the duplicate logic from `spot_reaction_mapper.py`, or remove `intent_selector.py` if `spot_reaction_mapper.py` is meant to be the sole source of truth going forward.

---

## Issues already fixed this session (for context / to sanity-check)

### 5. Offline pipeline silently zeroed out audio for `.mov` files

**File:** `test_emotions.py`, `extract_audio_features()`, lines 442–462

```python
442:        try:
443:            # Load audio from video. offset/duration let us slice out just the
444:            # portion of a longer source file that corresponds to one analysis window.
445:            #
446:            # librosa >=1.0 no longer falls back to audioread/ffmpeg when
447:            # soundfile/libsndfile can't parse the container (e.g. .mov). If the
448:            # direct load fails, convert to a temp .wav with ffmpeg ourselves and
449:            # retry once from that file.
450:            tmp_wav_path = None
451:            try:
452:                y, actual_sr = librosa.load(str(video_path), sr=sr, mono=True,
453:                                             offset=offset, duration=duration)
454:            except Exception as load_err:
455:                print(f"    librosa could not read '{video_path}' directly "
456:                      f"({load_err}); falling back to ffmpeg conversion...")
457:                tmp_wav_path = _ffmpeg_extract_wav(video_path, sr=sr)
458:                y, actual_sr = librosa.load(tmp_wav_path, sr=sr, mono=True,
459:                                             offset=offset, duration=duration)
460:            finally:
461:                if tmp_wav_path is not None and os.path.exists(tmp_wav_path):
462:                    os.unlink(tmp_wav_path)
```

**Root cause:** `libsndfile` (which `soundfile`/`librosa` use) has never supported `.mov` containers, and `librosa` 1.0 removed its old `audioread`/ffmpeg fallback entirely. **Fixed** by catching the failure and converting to `.wav` via a direct `ffmpeg` subprocess call (`_ffmpeg_extract_wav()`, defined just above this function) before retrying. Confirmed working — every window now extracts real audio instead of silently falling back to zeros.

### 6. Offline video segmentation used imprecise, non-deterministic seeking

**File:** `run_offline.py`, line 435

```python
435:            cap.set(cv2.CAP_PROP_POS_FRAMES, round(start_s * fps))
```

Was `cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)`. Millisecond-based seeking on compressed video converts timestamp→frame via container metadata before seeking to the nearest keyframe, and can land on a slightly different frame between runs. Switched to frame-index seeking, which skips that conversion step. **Note:** confirmed via checksum diagnostic (see below) that this fix worked completely — it wasn't the remaining source of run-to-run VA variance, which turned out to be issue #1 above.

A diagnostic (not meant to be permanent) is also in this file at lines 447–452, printing a SHA256 of each window's extracted clip — this is what let us prove video extraction was fully deterministic while VA output still varied, isolating the real bug to model loading:
```python
447:            # [DIAGNOSTIC] Checksum the extracted window clip so we can
451:            seg_hash = hashlib.sha256(Path(seg_path).read_bytes()).hexdigest()[:12]
452:            print(f"  [DIAGNOSTIC] {seg_path.name}: {n_frames_written} frames, sha256={seg_hash}", flush=True)
```
Safe to remove once the team is done investigating issue #1.

### 7. `CHECK_IN` treated "calm and positive" the same as "genuinely low-energy and positive"

**File:** `pipeline/spot_reaction_mapper.py`, constructor lines 44/57, branch at lines 197–213

```python
44:                 arousal_low_threshold: float = 0.07):
...
57:        self.arousal_low_threshold = arousal_low_threshold
...
197:            elif 'low-arousal' in va_lower:
198:                intent = "CHECK_IN"
199:                is_calm_positive = 'positive' in va_lower and arousal >= self.arousal_low_threshold
200:                if is_calm_positive:
201:                    # Calm and positive (neutral arousal, not truly low): more present.
202:                    speed_mult = 0.65
203:                    distance_mult = 0.9
204:                    duration_s = 1.5
205:                    explain = f"Calm-positive state (neutral arousal): checking in, more present (V={valence:.2f}, A={arousal:.2f})"
206:                else:
207:                    # Genuinely low arousal: gentle, standard CHECK_IN.
208:                    speed_mult = 0.5
209:                    distance_mult = 1.0
210:                    duration_s = 1.75
211:                    explain = f"Low-arousal state: checking in (V={valence:.2f}, A={arousal:.2f})"
212:                pose_mode = "VA_POSE" if use_va_pose else "REACTION_POSE"
213:                pose_name = "CHECK_IN" if not use_va_pose else "VA_POSE"
```

Since issue #3's label collapsing means both cases arrive here with the identical `"positive-low-arousal"` label, this uses the raw `arousal` float (already passed into `map_to_action()`) against a new configurable threshold to tell them apart, rather than changing the label itself. Confirmed working in a live run (`speed=0.65× dist=0.90×` correctly appeared for arousal ~0.11–0.12, above the 0.07 threshold).

---

## Summary for discussion

Issues #1–4 need team input before any code changes — #1 in particular is a correctness issue affecting both online and offline pipelines, not just a reproducibility annoyance, and is likely the highest priority. Issues #5–7 are implemented and were confirmed working in a local working copy; they are not yet in the GitHub repo and will be submitted separately for review.
