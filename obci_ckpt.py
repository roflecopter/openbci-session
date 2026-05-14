"""
%CKPT line parsing + intra-file gap inference for OpenBCI Cyton SD recordings.

Single source of truth for the gap-aware processing logic. Imported by:
- sd_convert.py (this repo) — public openbci-session pipeline
- py-qs-data/openbci_functions.py — private nightly pipeline (collect_bci)

The gap inference is the post-processor side of the firmware's per-block
recovery loop: when SD writes briefly fail (sniffer bumps, pSLC GC stalls),
the firmware drops some samples but emits a `%CKPT` heartbeat ~once per
minute carrying both `t=<ms_since_boot>` and `b=<block_counter>`. By
comparing wall-clock delta vs sample-index delta between consecutive
`%CKPT` lines we can locate where samples were dropped and zero-pad the
gap so downstream timestamps stay aligned.

Firmware reference: github.com/roflecopter/OpenBCI_Cyton_Library_SD
README "SD reliability and observability" section. The corresponding
firmware patches landed 2026-05-08 (initial %CKPT) and 2026-05-13
(extended-window recovery + new `x=N` field).
"""
from __future__ import annotations


def parse_ckpt_line(line):
    """Parse a `%CKPT t=<ms> b=<block> e=<errs> r=<retries> n=<reinits> o=<over> x=<extretries>` line.

    Returns dict with int values for any of {t, b, e, r, n, o, x} actually
    present. Returns None if the line is malformed (no `%CKPT ` prefix or
    no parseable key=value pairs). Tolerant of trailing/extra whitespace
    and unknown keys (silently ignored — forward-compat with future
    additions like a tunable-summary `T=<hash>` key per the firmware
    ROADMAP).

    Field semantics per firmware README:
      * t — millis() since cyton boot at emit time
      * b — blockCounter (which 512-byte SD block)
      * e — running count of card.writeData() failures
      * r — running count of writeStart retry attempts
      * n — running count of card.init() FAST-path recovery events
      * o — running count of block-write overruns
      * x — running count of EXTENDED-window recovery attempts (added 2026-05-13)
    """
    if not line.startswith('%CKPT '):
        return None
    payload = line[len('%CKPT '):].strip()
    if not payload:
        return None
    out = {}
    for kv in payload.split():
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        if k not in ('t', 'b', 'e', 'r', 'n', 'o', 'x'):
            continue  # unknown key — skip, forward-compat
        try:
            out[k] = int(v)
        except (TypeError, ValueError):
            continue
    return out if out else None


def compute_intra_file_gaps(ckpts, sf, jitter_tolerance_samples=5):
    """From a list of %CKPT dicts (with `sample_idx` set by the caller's
    process_file), infer intra-file sample gaps caused by SD-recovery
    sample drops.

    Each consecutive (ckpt_k, ckpt_{k+1}) pair gives us:
      * Δt_ms = ckpt_{k+1}['t'] - ckpt_k['t']  — wall-clock elapsed
      * Δsamples = ckpt_{k+1}['sample_idx'] - ckpt_k['sample_idx']  — recorded
      * expected = round(Δt_ms * sf / 1000)
      * missing  = expected - Δsamples

    If `missing > jitter_tolerance_samples` we record an intra-file gap of
    `missing` samples at `sample_idx = ckpt_{k+1}['sample_idx']` — i.e. JUST
    BEFORE the post-recovery sample stream resumes. (We don't know exactly
    when within the interval the recovery happened, but emitting the gap at
    the trailing CKPT is the closest we can pinpoint without per-`%E` line
    timing; epoch boundaries derived from samples-after-CKPT_{k+1} stay
    correct, and the gap is correctly attributed to "happened during the
    minute ending at the trailing CKPT".)

    Returns a list of dicts:
      [{
        'sample_idx': int,        # where to insert zero rows (in pre-gap-fill coords)
        'n_samples': int,         # how many zero rows to insert
        'gap_ms': int,            # implied DURATION OF THE GAP (= missing*1000/sf), NOT the full CKPT-to-CKPT Δt
        't_start_ms': int,        # leading-CKPT t_ms
        't_end_ms': int,          # trailing-CKPT t_ms (when gap was discovered)
        'errs_delta': int,        # b['e'] - a['e'] (informational)
        'extretries_delta': int,  # b['x'] - a['x'] (informational; 0 on pre-2026-05-13 firmware)
      }, ...]

    Skips:
      * pairs with missing keys (e.g. truncated %CKPT)
      * non-monotonic t (Δt_ms <= 0)
      * non-monotonic sample_idx (Δsamples < 0)
      * missing <= jitter_tolerance_samples (within ADC clock jitter)

    Defensive: if sf <= 0 the function returns [] — never divides by zero
    or fabricates gaps based on garbage.
    """
    gaps = []
    try:
        sf_f = float(sf)
    except (TypeError, ValueError):
        return gaps
    if sf_f <= 0:
        return gaps
    try:
        tol = int(jitter_tolerance_samples)
    except (TypeError, ValueError):
        tol = 0
    if tol < 0:
        tol = 0
    if not ckpts or len(ckpts) < 2:
        return gaps
    for k in range(len(ckpts) - 1):
        a, b = ckpts[k], ckpts[k + 1]
        if 't' not in a or 't' not in b:
            continue
        if 'sample_idx' not in a or 'sample_idx' not in b:
            continue
        dt_ms = b['t'] - a['t']
        if dt_ms <= 0:
            continue
        ds = b['sample_idx'] - a['sample_idx']
        if ds < 0:
            continue
        expected = int(round(dt_ms * sf_f / 1000.0))
        missing = expected - ds
        if missing <= tol:
            continue

        def _delta(key):
            if key in a and key in b:
                try:
                    d = int(b[key]) - int(a[key])
                    return d if d >= 0 else 0
                except (TypeError, ValueError):
                    return 0
            return 0

        gaps.append({
            'sample_idx': b['sample_idx'],
            'n_samples': missing,
            'gap_ms': int(round(missing * 1000.0 / sf_f)),
            't_start_ms': a['t'],
            't_end_ms': b['t'],
            'errs_delta': _delta('e'),
            'extretries_delta': _delta('x'),
        })
    return gaps
