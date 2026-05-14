"""tune_helpers.py — host-side helpers for the firmware tune protocol.

Pure functions: no serial I/O, no globals. session_start.py owns the
side-effecting plumbing (yml load, argparse, write to board.ser, ack
parsing). Splitting it out keeps unit tests cheap and the protocol
contract pinned to a single readable file.

Firmware contract: see OpenBCI_Cyton_Library_SD/examples/DefaultBoard/
SD_Card_Stuff.ino — the `Tune protocol` block at the top of the file is
the source of truth for key IDs, value widths, and valid ranges.

Wire format
-----------
Binary T command (sent live over serial):

    'T' <key_id:1B> <value bytes, LSB-first, count implied by key>

Firmware acks `TUNE OK <key_id>\\n$$$` on success, `TUNE FAIL <code>\\n$$$`
on failure (code: 1=unknown key, 2=value out of range).

Text %TUNE line (embedded in SESSION.TXT for auto-resume persistence):

    %TUNE max_resumes=25 ext_recovery_ms=8000 ext_chunk_ms=500 \\
          ckpt_interval_ms=60000 sd_write_timeout=1500

The firmware parses this in replaySessionFile() BEFORE feeding the rest
of the body through dispatch.
"""
import struct
from collections import OrderedDict


# Key ID -> (text-name, struct-pack-format, default-value, valid-range-incl)
# Order here is the canonical emit order for %TUNE text lines.
#
# Text key names MUST stay symmetric with the firmware C-variable names
# (`tuneMaxResumes` ↔ max_resumes, `tuneExtRecoveryWindowMs` ↔
# ext_recovery_window_ms, ...). A previous draft used the shorter
# ext_recovery_ms / ext_chunk_ms — review caught that asymmetry: anyone
# reading firmware code is likely to write the longer name on a host-side
# manual %TUNE edit, and the firmware silently skips unknown keys, so the
# wrong name on auto-resume would roll back to defaults. Renamed 2026-05-15.
TUNE_KEYS = OrderedDict([
    ('max_resumes',            (0x01, '<B', 25,    (1,    254))),
    ('ext_recovery_window_ms', (0x02, '<H', 8000,  (1,    60000))),
    ('ext_recovery_chunk_ms',  (0x03, '<H', 500,   (10,   5000))),
    ('ckpt_interval_ms',       (0x04, '<I', 60000, (1000, 3600000))),
    ('sd_write_timeout',       (0x05, '<H', 1500,  (100,  5000))),
])


def default_tune():
    """Return a fresh OrderedDict with all 5 keys at their firmware defaults."""
    return OrderedDict((name, meta[2]) for name, meta in TUNE_KEYS.items())


def parse_tune_arg(arg):
    """Parse a single `--tune key=value` argument into a (name, int) pair.

    Raises ValueError with a human-readable message on:
      - missing '='
      - unknown key
      - non-integer value (string-of-int OK)
      - out-of-range value (per TUNE_KEYS bounds)

    Cross-key constraints (chunk ≤ window) are NOT checked here — pair
    arguments are independent at the CLI level. Use merge_tune() at the
    end of all parsing to surface those.
    """
    if '=' not in arg:
        raise ValueError(f"--tune value must be key=value, got: {arg!r}")
    key, val_s = arg.split('=', 1)
    key = key.strip()
    val_s = val_s.strip()
    if key not in TUNE_KEYS:
        raise ValueError(
            f"unknown tune key: {key!r}; valid keys: {list(TUNE_KEYS.keys())}")
    val = _coerce_int(key, val_s)
    lo, hi = TUNE_KEYS[key][3]
    if val < lo or val > hi:
        raise ValueError(
            f"tune {key}={val} out of range [{lo}, {hi}]")
    return key, val


def _coerce_int(name, value):
    """Type-strict coercion for tune values.

    Rejects bool (True/False sneak through int() as 1/0), non-integer floats,
    None, lists, etc. Accepts: plain int, str-of-int (yml without quotes),
    int-valued float (e.g. yml `1500` parsed as Python int but `1500.0`
    parsed as float should also be accepted — both round to the same int).

    Centralised here so every entry point (parse_tune_arg, merge_tune)
    rejects the same set of bad values.
    """
    # Order matters: bool is a subclass of int, must reject before int check.
    if isinstance(value, bool):
        raise ValueError(
            f"tune {name} value must be integer, got bool: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"tune {name} value must be integer, got float: {value!r}")
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            raise ValueError(
                f"tune {name} value must be integer, got string: {value!r}")
    raise ValueError(
        f"tune {name} value must be integer, got {type(value).__name__}: "
        f"{value!r}")


def _check_cross_key(out):
    """Cross-key constraints the firmware enforces. Validate host-side
    BEFORE any T command goes out the door so a bad combination raises
    cleanly instead of as a mid-batch TUNE FAIL with partial state."""
    chunk = out.get('ext_recovery_chunk_ms')
    window = out.get('ext_recovery_window_ms')
    if chunk is not None and window is not None and chunk > window:
        raise ValueError(
            f"tune ext_recovery_chunk_ms={chunk} > "
            f"ext_recovery_window_ms={window}; chunk must be ≤ window or "
            f"the recovery window degenerates to a single attempt")


def merge_tune(*sources):
    """Merge tune sources in order, later wins.

    Each source must be a dict-like of {name: int}; non-dict raises
    ValueError (catches a malformed yml block like `tune: 5` BEFORE the
    board is touched). Unknown keys raise. Values are type-strict via
    _coerce_int — bool / non-integer float / non-numeric strings all
    rejected. Cross-key constraint chunk ≤ window enforced at the end.

    Returns a fresh OrderedDict with all 5 keys populated, defaults
    filled in for any source that omitted them.

    Typical call: `merge_tune(default_tune(), yml_block, cli_overrides)`.
    """
    out = default_tune()
    for src in sources:
        if src is None or src == {}:
            continue
        if not isinstance(src, dict):
            raise ValueError(
                f"tune source must be a dict/mapping, got "
                f"{type(src).__name__}: {src!r}")
        for k, v in src.items():
            if k not in TUNE_KEYS:
                raise ValueError(
                    f"unknown tune key: {k!r}; valid keys: "
                    f"{list(TUNE_KEYS.keys())}")
            vv = _coerce_int(k, v)
            lo, hi = TUNE_KEYS[k][3]
            if vv < lo or vv > hi:
                raise ValueError(
                    f"tune {k}={vv} out of range [{lo}, {hi}]")
            out[k] = vv
    _check_cross_key(out)
    return out


def build_t_command(key_name, value):
    """Build the binary 'T <key_id> <val bytes>' wire payload.

    Returns bytes ready to write to board.ser.

    Raises ValueError if key unknown or value out of range — same gates
    as parse_tune_arg, so callers can't smuggle a bad value past
    validation by going around the CLI path.
    """
    if key_name not in TUNE_KEYS:
        raise ValueError(f"unknown tune key: {key_name!r}")
    key_id, fmt, _default, (lo, hi) = TUNE_KEYS[key_name]
    if value < lo or value > hi:
        raise ValueError(
            f"tune {key_name}={value} out of range [{lo}, {hi}]")
    return b'T' + bytes([key_id]) + struct.pack(fmt, value)


def build_tune_text_line(tune):
    """Build a `%TUNE k=v k=v ...\\n` line from a {name: int} dict.

    Keys are emitted in TUNE_KEYS order (canonical) regardless of dict
    insertion order, so the resulting line is stable across calls and
    easy to diff. Returns bytes ready to embed in SESSION.TXT payload.

    Raises ValueError on unknown key (defensive — the firmware tolerates
    unknown keys silently, but we treat that as a host-side bug).
    """
    for k in tune:
        if k not in TUNE_KEYS:
            raise ValueError(f"unknown tune key: {k!r}")
    parts = []
    for name in TUNE_KEYS:
        if name in tune:
            parts.append(f'{name}={int(tune[name])}')
    return ('%TUNE ' + ' '.join(parts) + '\n').encode('ascii')
