"""sd_health.py — morning SD card health check.

Combines two independent signals into a single "is this card healthy
enough to record on tonight" verdict:

  1. live_write_test()    — write+fsync+verify a 1 MB pattern file on
                            the SD, measure per-block write latency and
                            count errors.
  2. parse_ckpt_summary() — parse last night's TXT for cumulative
                            %CKPT error counters (e/r/n/x/o) plus
                            wall-clock duration.

health_verdict() collapses both into HEALTHY / DEGRADING / DYING plus
a free-form notes string. persist_verdict() writes one row per morning
into a new SdHealth SQLite table so trends are visible (e.g. "p95
latency has crept from 80 ms → 240 ms over 30 nights → schedule
replacement"). Designed to be called from sd_convert.py's main block
after the BDF is written, or standalone via this file's CLI.

Firmware reference: github.com/roflecopter/OpenBCI_Cyton_Library_SD,
"SD reliability and observability" section in README.
"""
import argparse
import datetime
import os
import sqlite3
import statistics
import sys
import time
import uuid

from contextlib import closing

# %CKPT parser is the canonical implementation in sd_convert.py; import
# rather than duplicate so the two stay in lock-step on firmware-counter
# evolution.
try:
    from sd_convert import parse_ckpt_line
except ImportError:  # pragma: no cover — only happens if run from a
    # different cwd; the CLI path-prepends below cover the normal case.
    parse_ckpt_line = None


# ---------------------------------------------------------------------
# Verdict thresholds.
#
# Conservative starting values — the right calibration comes from a few
# weeks of "known-healthy" runs on the cards we actually use (Industrial
# 16 GB, Max Endurance 32 GB). Surface them as module-level constants so
# they can be tuned without editing decision logic; in time these can
# move into a YAML or even a per-card profile if cards diverge enough.
# ---------------------------------------------------------------------
DYING_LATENCY_P95_MS = 2000.0   # > the firmware's 1500 ms SD_WRITE_TIMEOUT
                                # ceiling — anything this slow guarantees
                                # the firmware's recovery cascade will
                                # fire on real writes. Raised from 1000
                                # 2026-05-15 after review: pSLC GC bursts
                                # on healthy cards hit 751 ms (CLAUDE.md);
                                # 1000 ms would label healthy-but-GC-burst
                                # cards DYING.
DEGRADING_LATENCY_P95_MS = 1000.0  # > observed pSLC GC burst ceiling
                                   # (751 ms) but below the firmware's
                                   # 1500 ms timeout — early warning that
                                   # the card is approaching the cliff.
DYING_THROUGHPUT_MBPS = 0.5     # cyton at 500 Hz writes ~2.5 KB/s of EEG
                                # but bursts at full SD speed during
                                # cache flush; sub-0.5 MB/s sustained is
                                # well below any healthy card.
DYING_VERIFY_ERRORS = 8         # >= 1 byte mismatch on read-back was too
                                # strict (a cosmic ray, a transient FS-
                                # cache glitch, or a sniffer-bumped probe
                                # would condemn a healthy card). 8 bytes
                                # within a 1 MB probe = real corruption.
                                # Single-byte mismatches push to DEGRADING
                                # via DEGRADING_VERIFY_ERRORS below.
DEGRADING_VERIFY_ERRORS = 1     # any non-zero verify mismatch is at
                                # least worth flagging for follow-up.
DYING_CKPT_EXTRETRIES = 50      # extended-window recovery fires per
                                # second of pSLC GC burst; 50 in a night
                                # = ~50 GC-burst events.
DYING_CKPT_ERRS = 50
DEGRADING_CKPT_EXTRETRIES = 5   # a handful of GC-burst recoveries is
                                # normal-ish on cheaper cards.
DEGRADING_CKPT_ERRS = 5
DEGRADING_CKPT_REINITS = 2      # a single benign reinit (e.g. one SD-
                                # sniffer bump during the night) is
                                # normal; two suggests intermittent
                                # contact or controller flakiness.

# Bumped when the schema or the shape of the JSON CLI output changes in
# a way downstream consumers (sessions.db trend tools, shell wrappers)
# need to notice. 1 = original. 2 = post-2026-05-15 review: added
# write/verify split, page-cache-busting, malformed_ckpt_count, UTC
# timestamps, UNKNOWN verdict.
SCHEMA_VERSION = 2


def live_write_test(sd_dir, payload_mb=1, chunk_kb=8, keep_probe=False):
    """Write+fsync+verify a `payload_mb` MB pattern file on `sd_dir`.

    Returns a dict:
      ok                  : bool — True when the test ran to completion
                                   ('bad payload/chunk size' returns
                                   ok=False with reason='config_invalid'
                                   so health_verdict can map to UNKNOWN
                                   instead of DYING.)
      reason              : str — 'ok' | 'config_invalid' | 'io_error'
      bytes_written       : int
      wall_time_s         : float — total (write + verify)
      write_time_s        : float — write phase only
      verify_time_s       : float — verify phase only
      mb_per_s            : float — composite (legacy, == bytes/wall_time_s)
      write_mb_per_s      : float — write-only throughput
      verify_mb_per_s     : float — verify-only throughput
      latency_p50_ms      : float — per-chunk fsync-included write latency
      latency_p95_ms      : float
      latency_max_ms      : float
      write_errors        : int  — exceptions during write/fsync
      verify_errors       : int  — bytes that read back wrong
      error               : str | None — first exception's message
      probe_path          : str — file path tested (deleted unless keep_probe)
      page_cache_busted   : bool — True if posix_fadvise(DONTNEED) was
                                    callable on this platform; False
                                    means the verify-read may have been
                                    served from page cache rather than
                                    re-fetched from the card.

    Latency is measured per `chunk_kb` chunk, including the fsync to
    the SD controller. That's what surfaces GC bursts — without fsync,
    Linux page cache absorbs the writes and the measurement degrades
    into "RAM speed". With fsync, the chunk-time distribution is the
    actual on-card behaviour.

    Verify-pass: on Linux we call `posix_fadvise(POSIX_FADV_DONTNEED)`
    on the probe file before re-opening for read, which forces the
    kernel to evict cached pages so the verify read genuinely re-fetches
    from the SD card. Without this, a flash-failing card whose volatile
    DRAM buffer is fine would pass verify (false-negative on health).

    Caveat — macOS: `posix_fadvise` is not available, and `os.fsync` on
    Darwin does not call `F_FULLFSYNC` (data may sit in the disk's own
    DRAM, not actually written to flash). On macOS the verify pass is
    weaker than on Linux. To get real device-level fsync on macOS, call
    `fcntl.fcntl(fd, fcntl.F_FULLFSYNC)` — not done here because the
    rig's primary processing host is Linux and the macOS path is a
    legacy compatibility surface, not the calibration platform.

    Designed to be cheap and recoverable: ~5–10 s on a healthy 16 GB
    Industrial-grade card, well under 60 s on anything not actively
    dying. Always tries to clean up the probe file, even on exception
    paths (unless `keep_probe=True` for debugging).
    """
    payload_bytes = int(payload_mb * 1024 * 1024)
    chunk_bytes = int(chunk_kb * 1024)
    if payload_bytes <= 0 or chunk_bytes <= 0 or chunk_bytes > payload_bytes:
        return {
            'ok': False,
            'reason': 'config_invalid',
            'bytes_written': 0,
            'wall_time_s': 0.0,
            'write_time_s': 0.0,
            'verify_time_s': 0.0,
            'mb_per_s': 0.0,
            'write_mb_per_s': 0.0,
            'verify_mb_per_s': 0.0,
            'latency_p50_ms': 0.0,
            'latency_p95_ms': 0.0,
            'latency_max_ms': 0.0,
            'write_errors': 0,
            'verify_errors': 0,
            'error': f'bad payload/chunk size: {payload_mb} MB / {chunk_kb} KB',
            'probe_path': '',
            'page_cache_busted': False,
        }

    # Pattern: full random payload, chunked. Earlier version used
    # `urandom(chunk) + rotation` which kept the byte-multiset identical
    # across chunks — a stuck-page failure returning a cached previous-
    # chunk page could pass verify by luck. 1 MB of urandom is trivial
    # in RAM (~10 ms cost) and eliminates that mode.
    raw_pattern = os.urandom(payload_bytes)
    n_chunks = payload_bytes // chunk_bytes
    pattern_chunks = [raw_pattern[i * chunk_bytes:(i + 1) * chunk_bytes]
                      for i in range(n_chunks)]

    probe_path = os.path.join(
        sd_dir, f'.sdhealth_probe_{os.getpid()}_{uuid.uuid4().hex[:8]}.bin')

    latencies_ms = []
    write_errors = 0
    verify_errors = 0
    err_msg = None
    bytes_written = 0
    page_cache_busted = False
    t_wall_start = time.monotonic()
    t_write_start = t_wall_start
    t_write_end = t_wall_start
    t_verify_start = t_wall_start
    t_verify_end = t_wall_start

    try:
        with open(probe_path, 'wb') as f:
            fd = f.fileno()
            for chunk in pattern_chunks:
                t0 = time.monotonic()
                try:
                    f.write(chunk)
                    f.flush()
                    os.fsync(fd)
                except OSError as e:
                    write_errors += 1
                    if err_msg is None:
                        err_msg = f'write: {e}'
                    # Don't bail — keep measuring the rest of the
                    # chunks so the report shows the failure pattern.
                else:
                    bytes_written += len(chunk)
                t1 = time.monotonic()
                latencies_ms.append((t1 - t0) * 1000.0)
            t_write_end = time.monotonic()

        # Page-cache busting BEFORE verify-read. On Linux,
        # posix_fadvise(POSIX_FADV_DONTNEED) tells the kernel to drop
        # cached pages for this file so the next read goes back to the
        # device. Without this, the verify on a removable SD could be
        # served from RAM and silently pass even on a failing card.
        if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
            try:
                with open(probe_path, 'rb') as f:
                    os.posix_fadvise(f.fileno(), 0, 0,
                                     os.POSIX_FADV_DONTNEED)
                page_cache_busted = True
            except OSError:
                # Some filesystems / mount options reject fadvise.
                # Continue — verify will still run, just possibly from cache.
                page_cache_busted = False

        # Read-back verify. Reopen with buffering=0 so each read() goes
        # to the OS layer directly (no Python-level buffer absorbing
        # consecutive chunks into one syscall).
        t_verify_start = time.monotonic()
        if write_errors == 0:
            with open(probe_path, 'rb', buffering=0) as f:
                for chunk in pattern_chunks:
                    got = f.read(len(chunk))
                    if got != chunk:
                        # Count mismatched bytes rather than chunks so
                        # a single bit-flip vs whole-chunk corruption is
                        # distinguishable in the trend data and the
                        # DEGRADING/DYING thresholds can be tuned in
                        # health_verdict.
                        verify_errors += sum(
                            1 for a, b in zip(got, chunk) if a != b)
                        if len(got) != len(chunk):
                            verify_errors += abs(len(got) - len(chunk))
        t_verify_end = time.monotonic()
    except OSError as e:
        err_msg = f'open: {e}'
    finally:
        if not keep_probe:
            try:
                os.remove(probe_path)
            except OSError:
                pass  # probe never created, or already gone

    wall_time_s = time.monotonic() - t_wall_start
    write_time_s = t_write_end - t_write_start
    verify_time_s = t_verify_end - t_verify_start
    bytes_mb = bytes_written / 1024.0 / 1024.0
    mb_per_s = (bytes_mb / wall_time_s) if wall_time_s > 0 else 0.0
    write_mb_per_s = (bytes_mb / write_time_s) if write_time_s > 0 else 0.0
    verify_mb_per_s = (bytes_mb / verify_time_s) if verify_time_s > 0 else 0.0

    if latencies_ms:
        # statistics.quantiles is python 3.8+; method='inclusive' so
        # we don't undershoot p95 on small samples like 128 chunks.
        try:
            p50 = statistics.median(latencies_ms)
        except statistics.StatisticsError:
            p50 = 0.0
        try:
            # quantiles needs ≥2 samples; we have 128 with 1 MB / 8 KB.
            qs = statistics.quantiles(latencies_ms, n=20, method='inclusive')
            p95 = qs[18]  # 95th percentile of the 19 cut points
        except statistics.StatisticsError:
            p95 = max(latencies_ms)
        p_max = max(latencies_ms)
    else:
        p50 = p95 = p_max = 0.0

    ran_to_completion = (write_errors == 0
                         and bytes_written == payload_bytes)
    return {
        'ok': ran_to_completion,
        'reason': 'ok' if ran_to_completion else 'io_error',
        'bytes_written': bytes_written,
        'wall_time_s': wall_time_s,
        'write_time_s': write_time_s,
        'verify_time_s': verify_time_s,
        'mb_per_s': mb_per_s,
        'write_mb_per_s': write_mb_per_s,
        'verify_mb_per_s': verify_mb_per_s,
        'latency_p50_ms': p50,
        'latency_p95_ms': p95,
        'latency_max_ms': p_max,
        'write_errors': write_errors,
        'verify_errors': verify_errors,
        'error': err_msg,
        'probe_path': probe_path,
        'page_cache_busted': page_cache_busted,
    }


def _empty_ckpt_summary():
    return {
        'ckpt_count': 0,
        'malformed_ckpt_count': 0,
        'first_t_ms': None,
        'last_t_ms': None,
        'duration_s': 0.0,
        'ckpt_errs': 0,
        'ckpt_retries': 0,
        'ckpt_reinits': 0,
        'ckpt_overruns': 0,
        'ckpt_extretries': 0,
    }


def parse_ckpt_summary(file_path):
    """Walk a TXT file and accumulate %CKPT counters.

    Returns dict:
      ckpt_count           : int — number of well-formed %CKPT lines parsed
      malformed_ckpt_count : int — `%CKPT ` lines parse_ckpt_line rejected
                                   (truncation, corruption, garbled keys).
                                   Non-zero is worth flagging in notes —
                                   it means the file has SD-corruption
                                   regions and the rest of the totals may
                                   underreport reality.
      first_t_ms           : int | None  — millis of first %CKPT (~boot)
      last_t_ms            : int | None
      duration_s           : float — (last_t - first_t) / 1000
      ckpt_errs            : int — cumulative `e=` across the session
      ckpt_retries         : int — cumulative `r=`
      ckpt_reinits         : int — cumulative `n=`
      ckpt_overruns        : int — cumulative `o=`
      ckpt_extretries      : int — cumulative `x=` (0 on pre-2026-05-13 fw)

    Pre-2026-05-08 firmware emits no %CKPT lines; the dict is returned
    fully zeroed and `ckpt_count = 0` — health_verdict() then leans on
    the live-write test alone.

    Reset-detection (added 2026-05-15 from review): the firmware's
    MAX_RESUMES auto-resume path soft-resets the cyton, which zeros all
    runtime counters. After a resume, subsequent %CKPT lines start fresh
    at e=0 r=0 etc. A naive max() across the whole file would under-
    report the night total (e.g. r=12 before halt, r=3 after resume →
    max=12, truth=15). We instead walk the sequence and detect resets
    (any counter going DOWN signals a resume), accumulating
    `prev_total + current_value` from there on.
    """
    if parse_ckpt_line is None:
        raise RuntimeError(
            'sd_convert.parse_ckpt_line not importable — '
            'add openbci-session to sys.path before calling parse_ckpt_summary')

    ckpts = []
    malformed = 0
    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('%CKPT '):
                parsed = parse_ckpt_line(line)
                if parsed is not None:
                    ckpts.append(parsed)
                else:
                    malformed += 1

    if not ckpts:
        result = _empty_ckpt_summary()
        result['malformed_ckpt_count'] = malformed
        return result

    ts = [c['t'] for c in ckpts if 't' in c]
    first_t = min(ts) if ts else None
    last_t = max(ts) if ts else None
    duration_s = ((last_t - first_t) / 1000.0) if (
        first_t is not None and last_t is not None) else 0.0

    # Sum-of-deltas walk for each counter. State per counter:
    #   carry      : sum of all completed (pre-reset) cumulative totals
    #   last_seen  : last counter value seen in this segment
    # When the next value drops below last_seen we treat that as a reset:
    #   carry += last_seen   (the pre-reset segment is now complete)
    #   last_seen = current_value (start of new segment)
    # At end of walk: total = carry + last_seen.
    def _cumulative(k):
        carry = 0
        last_seen = 0
        first_seen = False
        for c in ckpts:
            if k not in c:
                continue
            try:
                v = int(c[k])
            except (TypeError, ValueError):
                continue
            if not first_seen:
                last_seen = v
                first_seen = True
                continue
            if v < last_seen:
                # Reset detected — close out the previous segment and
                # start fresh.
                carry += last_seen
            last_seen = v
        return carry + last_seen if first_seen else 0

    return {
        'ckpt_count': len(ckpts),
        'malformed_ckpt_count': malformed,
        'first_t_ms': first_t,
        'last_t_ms': last_t,
        'duration_s': duration_s,
        'ckpt_errs': _cumulative('e'),
        'ckpt_retries': _cumulative('r'),
        'ckpt_reinits': _cumulative('n'),
        'ckpt_overruns': _cumulative('o'),
        'ckpt_extretries': _cumulative('x'),
    }


def health_verdict(live, ckpt_summary):
    """Combine live + CKPT signals into a single verdict.

    Returns (verdict, notes) where
      verdict ∈ {'HEALTHY','DEGRADING','DYING','UNKNOWN'}
      notes is a comma-joined list of the specific threshold trips
            (or 'clean' on HEALTHY, or the config error on UNKNOWN).

    Verdict ordering:
      UNKNOWN — live test could not run (bad config, no SD, etc.). Do
                NOT condemn the card based on this; the test never
                executed. Persisted so a morning sweep finds the entry,
                but downstream tooling should treat it as "no signal".
      DYING   — recording on this card tonight is unsafe.
      DEGRADING — record but watch trends.
      HEALTHY — go.

    DYING beats DEGRADING beats HEALTHY because the consequence is
    different — DEGRADING is "watch this card" (collect another night
    of data), DYING is "do not record on this card tonight" (swap it).

    A single verify_errors==1 or a single ckpt_reinits==1 are NOT
    instant DYING — they push to DEGRADING. Review caught the prior
    thresholds (verify=1 → DYING, reinits=1 → DEGRADING) producing
    false-positives on transient events (cosmic ray, single SD-sniffer
    bump). The dying thresholds now match real failure-mode signatures.
    """
    # UNKNOWN gate first — live test couldn't even attempt I/O.
    if live.get('reason') == 'config_invalid':
        return 'UNKNOWN', f"config_invalid: {live.get('error', 'unknown')}"

    reasons_dying = []
    reasons_degr = []

    verify_errs = live.get('verify_errors', 0)
    if verify_errs >= DYING_VERIFY_ERRORS:
        reasons_dying.append(f"verify_errors={verify_errs}")
    elif verify_errs >= DEGRADING_VERIFY_ERRORS:
        reasons_degr.append(f"verify_errors={verify_errs}")
    if live.get('write_errors', 0) > 0:
        reasons_dying.append(f"write_errors={live['write_errors']}")
    if not live.get('ok', False):
        # live_test_did_not_complete (and reason is NOT config_invalid,
        # since UNKNOWN handled that above) → I/O error during the run.
        reasons_dying.append('live_test_did_not_complete')
    p95 = live.get('latency_p95_ms', 0.0)
    if p95 >= DYING_LATENCY_P95_MS:
        reasons_dying.append(f"latency_p95={p95:.0f}ms")
    elif p95 >= DEGRADING_LATENCY_P95_MS:
        reasons_degr.append(f"latency_p95={p95:.0f}ms")
    # Write-only throughput is what matters for the recording
    # workload; verify-read can be near-instant if served from cache
    # (and even with cache-busting reads a different code path).
    mbps_write = live.get('write_mb_per_s', live.get('mb_per_s', 0.0))
    if 0 < mbps_write < DYING_THROUGHPUT_MBPS:
        reasons_dying.append(f"write_throughput={mbps_write:.2f}MB/s")

    extr = ckpt_summary.get('ckpt_extretries', 0)
    if extr >= DYING_CKPT_EXTRETRIES:
        reasons_dying.append(f"ckpt_extretries={extr}")
    elif extr >= DEGRADING_CKPT_EXTRETRIES:
        reasons_degr.append(f"ckpt_extretries={extr}")
    errs = ckpt_summary.get('ckpt_errs', 0)
    if errs >= DYING_CKPT_ERRS:
        reasons_dying.append(f"ckpt_errs={errs}")
    elif errs >= DEGRADING_CKPT_ERRS:
        reasons_degr.append(f"ckpt_errs={errs}")
    reinits = ckpt_summary.get('ckpt_reinits', 0)
    if reinits >= DEGRADING_CKPT_REINITS:
        # Multiple reinits = the firmware repeatedly dropped into the
        # slow card.init() recovery path. A single benign reinit (one
        # sniffer bump) doesn't move the verdict.
        reasons_degr.append(f"ckpt_reinits={reinits}")
    malformed = ckpt_summary.get('malformed_ckpt_count', 0)
    if malformed > 0:
        # Corrupt %CKPT lines mean part of the file is unreadable —
        # the rest of the totals may underreport reality. Worth a flag
        # but not on its own a verdict change (DEGRADING-side note).
        reasons_degr.append(f"malformed_ckpts={malformed}")

    if reasons_dying:
        return 'DYING', ', '.join(reasons_dying)
    if reasons_degr:
        return 'DEGRADING', ', '.join(reasons_degr)
    return 'HEALTHY', 'clean'


def persist_verdict(session_db, dts, sd_dir, txt_file, live, ckpt_summary,
                    verdict, notes):
    """Insert one row into the SdHealth table.

    Idempotent on schema (CREATE TABLE IF NOT EXISTS). The PRIMARY KEY
    is the timestamp `dts` so repeated runs for the same morning REPLACE
    rather than duplicate. Returns True on success, False on SQL error
    (logged via stderr — never raises, the verdict is already in hand
    and we don't want a DB hiccup to wipe it).
    """
    try:
        with closing(sqlite3.connect(session_db, timeout=10)) as con:
            with con:
                with closing(con.cursor()) as cur:
                    # Base schema (idempotent). New columns added 2026-05-15
                    # post-review (write_mb_per_s, verify_mb_per_s,
                    # malformed_ckpt_count, page_cache_busted, schema_version)
                    # are applied via ALTER TABLE below for upgrades-in-place.
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS SdHealth(
                            dts TEXT NOT NULL PRIMARY KEY,
                            sd_dir VARCHAR(256),
                            txt_file VARCHAR(256),
                            mb_per_s REAL,
                            latency_p50_ms REAL,
                            latency_p95_ms REAL,
                            latency_max_ms REAL,
                            write_errors INT,
                            verify_errors INT,
                            ckpt_count INT,
                            ckpt_errs INT,
                            ckpt_retries INT,
                            ckpt_reinits INT,
                            ckpt_overruns INT,
                            ckpt_extretries INT,
                            duration_s REAL,
                            verdict VARCHAR(16),
                            notes TEXT
                        )
                    """)
                    # Additive column upgrades — ignore "duplicate column"
                    # errors so the call is idempotent across runs.
                    for col_ddl in [
                        "ALTER TABLE SdHealth ADD COLUMN write_mb_per_s REAL",
                        "ALTER TABLE SdHealth ADD COLUMN verify_mb_per_s REAL",
                        "ALTER TABLE SdHealth ADD COLUMN malformed_ckpt_count INT",
                        "ALTER TABLE SdHealth ADD COLUMN page_cache_busted INT",
                        "ALTER TABLE SdHealth ADD COLUMN schema_version INT",
                    ]:
                        try:
                            cur.execute(col_ddl)
                        except sqlite3.OperationalError:
                            pass  # column already present
                    # Always store ISO-8601 UTC. Local-time wall clocks can
                    # drift on a fresh boot before NTP catches up; UTC keeps
                    # the row ordering stable even then.
                    if hasattr(dts, 'strftime'):
                        if dts.tzinfo is None:
                            dts_str = dts.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            dts_str = dts.astimezone(
                                datetime.timezone.utc).strftime(
                                    '%Y-%m-%d %H:%M:%SZ')
                    else:
                        dts_str = str(dts)
                    cur.execute("""
                        REPLACE INTO SdHealth (
                            dts, sd_dir, txt_file,
                            mb_per_s, write_mb_per_s, verify_mb_per_s,
                            latency_p50_ms, latency_p95_ms, latency_max_ms,
                            write_errors, verify_errors,
                            ckpt_count, malformed_ckpt_count,
                            ckpt_errs, ckpt_retries, ckpt_reinits,
                            ckpt_overruns, ckpt_extretries, duration_s,
                            verdict, notes,
                            page_cache_busted, schema_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        dts_str,
                        sd_dir, txt_file,
                        float(live.get('mb_per_s', 0.0)),
                        float(live.get('write_mb_per_s', 0.0)),
                        float(live.get('verify_mb_per_s', 0.0)),
                        float(live.get('latency_p50_ms', 0.0)),
                        float(live.get('latency_p95_ms', 0.0)),
                        float(live.get('latency_max_ms', 0.0)),
                        int(live.get('write_errors', 0)),
                        int(live.get('verify_errors', 0)),
                        int(ckpt_summary.get('ckpt_count', 0)),
                        int(ckpt_summary.get('malformed_ckpt_count', 0)),
                        int(ckpt_summary.get('ckpt_errs', 0)),
                        int(ckpt_summary.get('ckpt_retries', 0)),
                        int(ckpt_summary.get('ckpt_reinits', 0)),
                        int(ckpt_summary.get('ckpt_overruns', 0)),
                        int(ckpt_summary.get('ckpt_extretries', 0)),
                        float(ckpt_summary.get('duration_s', 0.0)),
                        verdict, notes,
                        1 if live.get('page_cache_busted', False) else 0,
                        SCHEMA_VERSION,
                    ))
        return True
    except sqlite3.Error as e:
        print(f'WARN: SdHealth persist failed: {e}', file=sys.stderr)
        return False


def format_report(verdict, notes, live, ckpt_summary, sd_dir, txt_file):
    """Human-readable one-screen summary for the morning console log."""
    lines = [
        f'=== SD Health: {verdict} ===',
        f'  reasons: {notes}',
        f'  sd_dir:  {sd_dir}',
        f'  txt:     {txt_file or "(no overnight TXT — live-write only)"}',
        '  --- live write test ---',
        f'  ok={live.get("ok", False)} '
        f'bytes={live.get("bytes_written", 0)} '
        f'wall={live.get("wall_time_s", 0.0):.2f}s',
        f'  throughput: write={live.get("write_mb_per_s", 0.0):.2f} MB/s '
        f'verify={live.get("verify_mb_per_s", 0.0):.2f} MB/s',
        f'  latency p50/p95/max = '
        f'{live.get("latency_p50_ms", 0.0):.1f} / '
        f'{live.get("latency_p95_ms", 0.0):.1f} / '
        f'{live.get("latency_max_ms", 0.0):.1f} ms',
        f'  write_errors={live.get("write_errors", 0)} '
        f'verify_errors={live.get("verify_errors", 0)} '
        f'page_cache_busted={live.get("page_cache_busted", False)}',
    ]
    if live.get('error'):
        lines.append(f'  live test error: {live["error"]}')
    if ckpt_summary.get('ckpt_count', 0) > 0:
        lines.append('  --- last night %CKPT summary ---')
        lines.append(
            f'  duration={ckpt_summary["duration_s"]/3600:.2f}h '
            f'ckpts={ckpt_summary["ckpt_count"]}'
            + (f' malformed={ckpt_summary["malformed_ckpt_count"]}'
               if ckpt_summary.get('malformed_ckpt_count', 0) else ''))
        lines.append(
            f'  e={ckpt_summary["ckpt_errs"]} '
            f'r={ckpt_summary["ckpt_retries"]} '
            f'n={ckpt_summary["ckpt_reinits"]} '
            f'o={ckpt_summary["ckpt_overruns"]} '
            f'x={ckpt_summary["ckpt_extretries"]} '
            f'(cumulative with auto-resume reset detection)')
    return '\n'.join(lines)


def run_health_check(sd_dir, txt_file=None, session_db=None,
                     payload_mb=1, chunk_kb=8):
    """End-to-end orchestration: live test + (optional) CKPT parse +
    verdict + (optional) persist + report string.

    Returns (verdict, notes, live, ckpt_summary, report_string).
    The persist step is skipped silently when session_db is None.
    """
    live = live_write_test(sd_dir, payload_mb=payload_mb, chunk_kb=chunk_kb)

    if txt_file and os.path.exists(txt_file):
        ckpt = parse_ckpt_summary(txt_file)
    else:
        ckpt = _empty_ckpt_summary()

    verdict, notes = health_verdict(live, ckpt)

    if session_db:
        # UTC timestamp guards against pre-NTP-sync local clock drift on
        # a fresh boot — otherwise two morning runs minutes apart could
        # land in the wrong order. ISO-8601 stays string-sortable.
        persist_verdict(session_db,
                        datetime.datetime.now(datetime.timezone.utc),
                        sd_dir,
                        os.path.basename(txt_file) if txt_file else '',
                        live, ckpt, verdict, notes)

    report = format_report(verdict, notes, live, ckpt, sd_dir, txt_file)
    return verdict, notes, live, ckpt, report


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _main():
    ap = argparse.ArgumentParser(description=(
        'Run the morning SD health check: live write-verify + parse '
        "last night's %CKPT counters, emit verdict, optionally persist."))
    ap.add_argument('--sd-dir', required=True,
                    help='SD card mount point (must be writable)')
    ap.add_argument('--txt', default=None,
                    help='Last night TXT to parse for %%CKPT counters '
                         '(optional — skipped if absent)')
    ap.add_argument('--session-db', default=None,
                    help='Path to sessions.db for verdict persistence '
                         '(optional — verdict still printed if omitted)')
    ap.add_argument('--payload-mb', type=int, default=1,
                    help='Size of the live-write probe (default 1 MB)')
    ap.add_argument('--chunk-kb', type=int, default=8,
                    help='Per-chunk size for latency sampling (default 8 KB)')
    ap.add_argument('--json', action='store_true',
                    help='Emit machine-readable JSON instead of a report')
    args = ap.parse_args()

    verdict, notes, live, ckpt, report = run_health_check(
        args.sd_dir, txt_file=args.txt, session_db=args.session_db,
        payload_mb=args.payload_mb, chunk_kb=args.chunk_kb)

    if args.json:
        import json as _json
        out = {
            'schema_version': SCHEMA_VERSION,
            'verdict': verdict,
            'notes': notes,
            'live': live,
            'ckpt_summary': ckpt,
            'sd_dir': args.sd_dir,
            'txt': args.txt,
        }
        print(_json.dumps(out, indent=2, default=str))
    else:
        print(report)

    # Exit code mirrors verdict so a morning shell wrapper can branch on
    # it. UNKNOWN (live test could not run, e.g. bad config) → 3 so
    # `if rc -ge 1` still trips for non-healthy. Anything else → 4.
    sys.exit({'HEALTHY': 0, 'DEGRADING': 1, 'DYING': 2,
              'UNKNOWN': 3}.get(verdict, 4))


if __name__ == '__main__':
    # Allow running from any cwd: prepend the script's dir to sys.path
    # so the sd_convert import above resolves.
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    # Re-import parse_ckpt_line now that path is set, if it was None.
    if parse_ckpt_line is None:
        from sd_convert import parse_ckpt_line as _pcl  # noqa: F401
        globals()['parse_ckpt_line'] = _pcl
    _main()
