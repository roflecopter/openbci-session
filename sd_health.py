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
DYING_LATENCY_P95_MS = 1000.0   # ≥ the firmware's 1500 ms SD_WRITE_TIMEOUT
                                # ceiling — anything this slow risks
                                # timing out a real write.
DEGRADING_LATENCY_P95_MS = 300.0  # 4x the observed healthy p95 (~80 ms)
                                  # — early warning, not failure.
DYING_THROUGHPUT_MBPS = 0.5     # cyton at 500 Hz writes ~2.5 KB/s of EEG
                                # but bursts at full SD speed during
                                # cache flush; sub-0.5 MB/s sustained is
                                # well below any healthy card.
DYING_VERIFY_ERRORS = 1         # ANY bit-flip on read-back is fatal.
DYING_CKPT_EXTRETRIES = 50      # extended-window recovery fires per
                                # second of pSLC GC burst; 50 in a night
                                # = ~50 GC-burst events.
DYING_CKPT_ERRS = 50
DEGRADING_CKPT_EXTRETRIES = 5   # a handful of GC-burst recoveries is
                                # normal-ish on cheaper cards.
DEGRADING_CKPT_ERRS = 5


def live_write_test(sd_dir, payload_mb=1, chunk_kb=8, keep_probe=False):
    """Write+fsync+verify a `payload_mb` MB pattern file on `sd_dir`.

    Returns a dict:
      ok                : bool — True when the test ran to completion
      bytes_written     : int
      wall_time_s       : float
      mb_per_s          : float
      latency_p50_ms    : float — per-chunk fsync-included latency
      latency_p95_ms    : float
      latency_max_ms    : float
      write_errors      : int  — exceptions during write/fsync
      verify_errors     : int  — bytes that read back wrong
      error             : str | None — first exception's message
      probe_path        : str — file path tested (deleted unless keep_probe)

    Latency is measured per `chunk_kb` chunk, including the fsync to
    the SD controller. That's what surfaces GC bursts — without fsync,
    Linux page cache absorbs the writes and the measurement degrades
    into "RAM speed". With fsync, the chunk-time distribution is the
    actual on-card behaviour.

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
            'bytes_written': 0,
            'wall_time_s': 0.0,
            'mb_per_s': 0.0,
            'latency_p50_ms': 0.0,
            'latency_p95_ms': 0.0,
            'latency_max_ms': 0.0,
            'write_errors': 0,
            'verify_errors': 0,
            'error': f'bad payload/chunk size: {payload_mb} MB / {chunk_kb} KB',
            'probe_path': '',
        }

    # Pattern: deterministic-but-non-uniform so a card with a single
    # stuck bit can't pass via "everything reads back as 0xFF". Uses
    # os.urandom for the seed-pattern then a cheap rotation — keeps the
    # whole probe under 1 MB of RAM and reproducible-enough that a
    # verify mismatch points at the card, not at us.
    rng_seed = os.urandom(chunk_bytes)
    pattern_chunks = []
    for i in range(payload_bytes // chunk_bytes):
        # rotate the seed by i bytes each chunk so a stuck-page failure
        # mode surfaces as a non-zero verify_errors rather than passing
        # because two chunks happen to be identical.
        rot = i % chunk_bytes
        pattern_chunks.append(rng_seed[rot:] + rng_seed[:rot])

    probe_path = os.path.join(
        sd_dir, f'.sdhealth_probe_{os.getpid()}_{uuid.uuid4().hex[:8]}.bin')

    latencies_ms = []
    write_errors = 0
    verify_errors = 0
    err_msg = None
    bytes_written = 0
    t_wall_start = time.monotonic()

    try:
        # Open with O_SYNC isn't portable (some macOS filesystems treat
        # it loosely), so we do explicit fsync per chunk for repeatable
        # latency semantics across Linux+Mac+whatever.
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

        # Read-back verify. Reopen so the OS doesn't serve from any
        # write-side buffers we still hold. fsync above already pushed
        # everything to the SD, but a fresh fd is the cleanest.
        if write_errors == 0:
            with open(probe_path, 'rb') as f:
                for chunk in pattern_chunks:
                    got = f.read(len(chunk))
                    if got != chunk:
                        # count mismatched bytes rather than chunks so
                        # one bit-flip vs whole-chunk corruption is
                        # distinguishable in the trend data.
                        verify_errors += sum(
                            1 for a, b in zip(got, chunk) if a != b)
                        if len(got) != len(chunk):
                            verify_errors += abs(len(got) - len(chunk))
    except OSError as e:
        err_msg = f'open: {e}'
    finally:
        if not keep_probe:
            try:
                os.remove(probe_path)
            except OSError:
                pass  # probe never created, or already gone

    wall_time_s = time.monotonic() - t_wall_start
    mb_per_s = (bytes_written / 1024.0 / 1024.0) / wall_time_s \
        if wall_time_s > 0 else 0.0

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

    return {
        'ok': write_errors == 0 and bytes_written == payload_bytes,
        'bytes_written': bytes_written,
        'wall_time_s': wall_time_s,
        'mb_per_s': mb_per_s,
        'latency_p50_ms': p50,
        'latency_p95_ms': p95,
        'latency_max_ms': p_max,
        'write_errors': write_errors,
        'verify_errors': verify_errors,
        'error': err_msg,
        'probe_path': probe_path,
    }


def parse_ckpt_summary(file_path):
    """Walk a TXT file and accumulate %CKPT counters.

    Returns dict:
      ckpt_count       : int — number of %CKPT lines parsed
      first_t_ms       : int | None  — millis of first %CKPT (= ~boot)
      last_t_ms        : int | None
      duration_s       : float — (last_t - first_t) / 1000
      ckpt_errs        : int — last %CKPT's e= (cumulative)
      ckpt_retries     : int — last %CKPT's r=
      ckpt_reinits     : int — last %CKPT's n=
      ckpt_overruns    : int — last %CKPT's o=
      ckpt_extretries  : int — last %CKPT's x= (0 on pre-2026-05-13 firmware)

    Pre-2026-05-08 firmware emits no %CKPT lines; we return zeroed
    counters and a `ckpt_count = 0` flag — health_verdict() then leans
    on the live-write test alone.

    Defensive: silently tolerates malformed %CKPT lines (parse_ckpt_line
    returns None and we skip), missing keys, non-monotonic t (last_t
    becomes the max seen rather than the literal last). File-IO errors
    propagate so the caller can mark the verdict as 'unknown'.
    """
    if parse_ckpt_line is None:
        raise RuntimeError(
            'sd_convert.parse_ckpt_line not importable — '
            'add openbci-session to sys.path before calling parse_ckpt_summary')

    ckpts = []
    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('%CKPT '):
                parsed = parse_ckpt_line(line)
                if parsed is not None:
                    ckpts.append(parsed)

    if not ckpts:
        return {
            'ckpt_count': 0,
            'first_t_ms': None,
            'last_t_ms': None,
            'duration_s': 0.0,
            'ckpt_errs': 0,
            'ckpt_retries': 0,
            'ckpt_reinits': 0,
            'ckpt_overruns': 0,
            'ckpt_extretries': 0,
        }

    ts = [c['t'] for c in ckpts if 't' in c]
    first_t = min(ts) if ts else None
    last_t = max(ts) if ts else None
    duration_s = ((last_t - first_t) / 1000.0) if (
        first_t is not None and last_t is not None) else 0.0

    # Counters are cumulative since boot; the last %CKPT's value is the
    # session total. Take max() rather than last to defend against a
    # mid-file %CKPT with corrupt counter value (e.g. an SD glitch that
    # smeared a digit) — counters are monotonic non-decreasing, so max
    # is always the "real" value.
    def _max_key(k):
        return max((c[k] for c in ckpts if k in c), default=0)

    return {
        'ckpt_count': len(ckpts),
        'first_t_ms': first_t,
        'last_t_ms': last_t,
        'duration_s': duration_s,
        'ckpt_errs': _max_key('e'),
        'ckpt_retries': _max_key('r'),
        'ckpt_reinits': _max_key('n'),
        'ckpt_overruns': _max_key('o'),
        'ckpt_extretries': _max_key('x'),
    }


def health_verdict(live, ckpt_summary):
    """Combine live + CKPT signals into a single verdict.

    Returns (verdict, notes) where verdict ∈ {'HEALTHY','DEGRADING','DYING'}
    and notes is a comma-joined list of the specific threshold trips
    (or 'clean' on HEALTHY). Both signals are evaluated independently:
    *any* DYING signal pushes verdict to DYING; absent DYING, *any*
    DEGRADING pushes to DEGRADING; absent both, HEALTHY.

    DYING beats DEGRADING because the consequence is different —
    DEGRADING is "watch this card" (collect another night of data),
    DYING is "do not record on this card tonight" (swap it).
    """
    reasons_dying = []
    reasons_degr = []

    if live.get('verify_errors', 0) >= DYING_VERIFY_ERRORS:
        reasons_dying.append(f"verify_errors={live['verify_errors']}")
    if live.get('write_errors', 0) > 0:
        reasons_dying.append(f"write_errors={live['write_errors']}")
    if not live.get('ok', False):
        reasons_dying.append('live_test_did_not_complete')
    p95 = live.get('latency_p95_ms', 0.0)
    if p95 >= DYING_LATENCY_P95_MS:
        reasons_dying.append(f"latency_p95={p95:.0f}ms")
    elif p95 >= DEGRADING_LATENCY_P95_MS:
        reasons_degr.append(f"latency_p95={p95:.0f}ms")
    mbps = live.get('mb_per_s', 0.0)
    if 0 < mbps < DYING_THROUGHPUT_MBPS:
        reasons_dying.append(f"throughput={mbps:.2f}MB/s")

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
    if reinits > 0:
        # any reinit = the firmware had to drop into the slow
        # card.init() recovery path. Not necessarily fatal but always
        # worth flagging.
        reasons_degr.append(f"ckpt_reinits={reinits}")

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
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS SdHealth(
                            dts datetime NOT NULL PRIMARY KEY,
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
                    cur.execute("""
                        REPLACE INTO SdHealth (
                            dts, sd_dir, txt_file,
                            mb_per_s, latency_p50_ms, latency_p95_ms, latency_max_ms,
                            write_errors, verify_errors,
                            ckpt_count, ckpt_errs, ckpt_retries, ckpt_reinits,
                            ckpt_overruns, ckpt_extretries, duration_s,
                            verdict, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        dts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(dts, 'strftime') else str(dts),
                        sd_dir, txt_file,
                        float(live.get('mb_per_s', 0.0)),
                        float(live.get('latency_p50_ms', 0.0)),
                        float(live.get('latency_p95_ms', 0.0)),
                        float(live.get('latency_max_ms', 0.0)),
                        int(live.get('write_errors', 0)),
                        int(live.get('verify_errors', 0)),
                        int(ckpt_summary.get('ckpt_count', 0)),
                        int(ckpt_summary.get('ckpt_errs', 0)),
                        int(ckpt_summary.get('ckpt_retries', 0)),
                        int(ckpt_summary.get('ckpt_reinits', 0)),
                        int(ckpt_summary.get('ckpt_overruns', 0)),
                        int(ckpt_summary.get('ckpt_extretries', 0)),
                        float(ckpt_summary.get('duration_s', 0.0)),
                        verdict, notes,
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
        f'  ok={live.get("ok", False)} bytes={live.get("bytes_written", 0)} '
        f'time={live.get("wall_time_s", 0.0):.2f}s '
        f'throughput={live.get("mb_per_s", 0.0):.2f} MB/s',
        f'  latency p50/p95/max = '
        f'{live.get("latency_p50_ms", 0.0):.1f} / '
        f'{live.get("latency_p95_ms", 0.0):.1f} / '
        f'{live.get("latency_max_ms", 0.0):.1f} ms',
        f'  write_errors={live.get("write_errors", 0)} '
        f'verify_errors={live.get("verify_errors", 0)}',
    ]
    if live.get('error'):
        lines.append(f'  live test error: {live["error"]}')
    if ckpt_summary.get('ckpt_count', 0) > 0:
        lines.append('  --- last night %CKPT summary ---')
        lines.append(
            f'  duration={ckpt_summary["duration_s"]/3600:.2f}h '
            f'ckpts={ckpt_summary["ckpt_count"]}')
        lines.append(
            f'  e={ckpt_summary["ckpt_errs"]} '
            f'r={ckpt_summary["ckpt_retries"]} '
            f'n={ckpt_summary["ckpt_reinits"]} '
            f'o={ckpt_summary["ckpt_overruns"]} '
            f'x={ckpt_summary["ckpt_extretries"]}')
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
        ckpt = {
            'ckpt_count': 0,
            'first_t_ms': None,
            'last_t_ms': None,
            'duration_s': 0.0,
            'ckpt_errs': 0,
            'ckpt_retries': 0,
            'ckpt_reinits': 0,
            'ckpt_overruns': 0,
            'ckpt_extretries': 0,
        }

    verdict, notes = health_verdict(live, ckpt)

    if session_db:
        persist_verdict(session_db, datetime.datetime.now(), sd_dir,
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

    # Exit code mirrors verdict so a morning shell wrapper can branch on it.
    sys.exit({'HEALTHY': 0, 'DEGRADING': 1, 'DYING': 2}.get(verdict, 3))


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
