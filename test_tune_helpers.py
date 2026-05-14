"""Tests for tune_helpers.py — protocol contract pinned at host side."""
import os
import sys
import unittest

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import tune_helpers as th  # noqa: E402


class TestParseTuneArg(unittest.TestCase):
    def test_happy(self):
        self.assertEqual(th.parse_tune_arg('max_resumes=10'),
                         ('max_resumes', 10))

    def test_strips_whitespace(self):
        self.assertEqual(th.parse_tune_arg('  max_resumes = 10 '),
                         ('max_resumes', 10))

    def test_missing_equals(self):
        with self.assertRaisesRegex(ValueError, 'key=value'):
            th.parse_tune_arg('max_resumes10')

    def test_unknown_key(self):
        with self.assertRaisesRegex(ValueError, 'unknown tune key'):
            th.parse_tune_arg('nope=1')

    def test_non_int_value(self):
        with self.assertRaisesRegex(ValueError, 'integer'):
            th.parse_tune_arg('max_resumes=abc')

    def test_out_of_range_low(self):
        with self.assertRaisesRegex(ValueError, 'out of range'):
            th.parse_tune_arg('max_resumes=0')

    def test_out_of_range_high(self):
        with self.assertRaisesRegex(ValueError, 'out of range'):
            th.parse_tune_arg('max_resumes=255')

    def test_range_inclusive_at_boundaries(self):
        self.assertEqual(th.parse_tune_arg('max_resumes=1')[1], 1)
        self.assertEqual(th.parse_tune_arg('max_resumes=254')[1], 254)
        self.assertEqual(th.parse_tune_arg('sd_write_timeout=100')[1], 100)
        self.assertEqual(th.parse_tune_arg('sd_write_timeout=5000')[1], 5000)


class TestDefaults(unittest.TestCase):
    def test_default_tune_matches_firmware(self):
        d = th.default_tune()
        self.assertEqual(d['max_resumes'], 25)
        self.assertEqual(d['ext_recovery_window_ms'], 8000)
        self.assertEqual(d['ext_recovery_chunk_ms'], 500)
        self.assertEqual(d['ckpt_interval_ms'], 60000)
        self.assertEqual(d['sd_write_timeout'], 1500)


class TestMergeTune(unittest.TestCase):
    def test_no_sources_returns_defaults(self):
        self.assertEqual(th.merge_tune(), th.default_tune())

    def test_later_wins(self):
        result = th.merge_tune({'max_resumes': 10}, {'max_resumes': 20})
        self.assertEqual(result['max_resumes'], 20)

    def test_other_keys_keep_defaults(self):
        result = th.merge_tune({'max_resumes': 10})
        self.assertEqual(result['ext_recovery_window_ms'], 8000)

    def test_unknown_key_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown tune key'):
            th.merge_tune({'bogus': 1})

    def test_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError, 'out of range'):
            th.merge_tune({'max_resumes': 9999})

    def test_str_value_coerced(self):
        # yml might give us strings if user quoted the number — coerce.
        r = th.merge_tune({'max_resumes': '15'})
        self.assertEqual(r['max_resumes'], 15)


class TestBuildTCommand(unittest.TestCase):
    def test_max_resumes(self):
        # 'T' (0x54), key 0x01, value 25 = 0x19
        self.assertEqual(th.build_t_command('max_resumes', 25),
                         b'T\x01\x19')

    def test_ext_recovery_window_ms_lsb_first(self):
        # value 8000 = 0x1F40 → LSB first 0x40 0x1F
        self.assertEqual(th.build_t_command('ext_recovery_window_ms', 8000),
                         b'T\x02\x40\x1F')

    def test_ckpt_interval_ms_uint32(self):
        # 60000 = 0x0000_EA60 → LSB first 0x60 0xEA 0x00 0x00
        self.assertEqual(th.build_t_command('ckpt_interval_ms', 60000),
                         b'T\x04\x60\xEA\x00\x00')

    def test_sd_write_timeout_uint16(self):
        # 1500 = 0x05DC → 0xDC 0x05
        self.assertEqual(th.build_t_command('sd_write_timeout', 1500),
                         b'T\x05\xDC\x05')

    def test_unknown_key(self):
        with self.assertRaisesRegex(ValueError, 'unknown tune key'):
            th.build_t_command('nope', 1)

    def test_out_of_range(self):
        with self.assertRaisesRegex(ValueError, 'out of range'):
            th.build_t_command('max_resumes', 999)


class TestBuildTuneTextLine(unittest.TestCase):
    def test_canonical_order(self):
        # Even if user passes in different order, output is canonical.
        line = th.build_tune_text_line({
            'sd_write_timeout': 1500,
            'max_resumes': 25,
            'ckpt_interval_ms': 60000,
            'ext_recovery_chunk_ms': 500,
            'ext_recovery_window_ms': 8000,
        })
        self.assertEqual(
            line,
            b'%TUNE max_resumes=25 ext_recovery_window_ms=8000 ext_recovery_chunk_ms=500 '
            b'ckpt_interval_ms=60000 sd_write_timeout=1500\n')

    def test_partial(self):
        line = th.build_tune_text_line({'max_resumes': 10})
        self.assertEqual(line, b'%TUNE max_resumes=10\n')

    def test_unknown_key(self):
        with self.assertRaisesRegex(ValueError, 'unknown tune key'):
            th.build_tune_text_line({'bogus': 1})


class TestStrictCoercion(unittest.TestCase):
    """Type-strict coercion in merge_tune / parse_tune_arg — review caught
    these slipping through int() silently."""

    def test_bool_true_rejected(self):
        # yaml `max_resumes: true` parses as Python True → int(True)==1 → in
        # range; would silently accept "true" as "1". Reject explicitly.
        with self.assertRaisesRegex(ValueError, 'bool'):
            th.merge_tune({'max_resumes': True})

    def test_bool_false_rejected(self):
        with self.assertRaisesRegex(ValueError, 'bool'):
            th.merge_tune({'max_resumes': False})

    def test_float_non_integer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'float'):
            th.merge_tune({'max_resumes': 10.7})

    def test_float_integer_accepted(self):
        # yml may parse `25` as float in some loaders; if it's int-valued
        # (no fractional part) accept.
        r = th.merge_tune({'max_resumes': 25.0})
        self.assertEqual(r['max_resumes'], 25)

    def test_none_rejected(self):
        # yml-empty key produces None — rejected with a clear message.
        with self.assertRaisesRegex(ValueError, 'NoneType'):
            th.merge_tune({'max_resumes': None})

    def test_list_value_rejected(self):
        with self.assertRaisesRegex(ValueError, 'list'):
            th.merge_tune({'max_resumes': [25]})

    def test_string_with_garbage_rejected(self):
        with self.assertRaisesRegex(ValueError, 'string'):
            th.merge_tune({'max_resumes': '10abc'})


class TestMergeNonDictSource(unittest.TestCase):
    """A malformed yml `tune: 5` returns a non-dict that previously raised
    AttributeError on src.items() — opaque trace. Now raises ValueError."""

    def test_int_source(self):
        with self.assertRaisesRegex(ValueError, 'must be a dict'):
            th.merge_tune(5)

    def test_list_source(self):
        with self.assertRaisesRegex(ValueError, 'must be a dict'):
            th.merge_tune(['max_resumes', 25])

    def test_string_source(self):
        with self.assertRaisesRegex(ValueError, 'must be a dict'):
            th.merge_tune('max_resumes=25')

    def test_none_source_ok(self):
        # None means "no override" — should be silently skipped, not
        # raise. Tests the existing behaviour stays.
        self.assertEqual(th.merge_tune(None), th.default_tune())

    def test_empty_dict_ok(self):
        self.assertEqual(th.merge_tune({}), th.default_tune())


class TestCrossKeyConstraint(unittest.TestCase):
    """chunk ≤ window is a firmware-side constraint (applyTune rejects
    chunk > window). Mirror it host-side so a bad combination fails
    cleanly BEFORE the board is touched, not mid-batch."""

    def test_chunk_below_window_ok(self):
        r = th.merge_tune({'ext_recovery_window_ms': 4000,
                           'ext_recovery_chunk_ms': 500})
        self.assertEqual(r['ext_recovery_chunk_ms'], 500)

    def test_chunk_equal_window_ok(self):
        r = th.merge_tune({'ext_recovery_window_ms': 4000,
                           'ext_recovery_chunk_ms': 4000})
        self.assertEqual(r['ext_recovery_chunk_ms'], 4000)

    def test_chunk_exceeds_window_rejected(self):
        with self.assertRaisesRegex(ValueError, 'chunk_ms=5000'):
            th.merge_tune({'ext_recovery_window_ms': 3000,
                           'ext_recovery_chunk_ms': 5000})

    def test_defaults_satisfy_constraint(self):
        # 500 ≤ 8000 by design; sanity-pin so changing defaults can't
        # silently introduce an invalid baseline.
        d = th.default_tune()
        self.assertLessEqual(d['ext_recovery_chunk_ms'],
                             d['ext_recovery_window_ms'])


class TestEndToEnd(unittest.TestCase):
    """Quick contract sanity: defaults + text line + binary cmd are
    consistent and parseable per the firmware's documented format."""

    def test_text_line_roundtrip_through_defaults(self):
        # The %TUNE line emitted at defaults should parse back to the
        # same key=value pairs the firmware applies via applyTune.
        d = th.default_tune()
        line = th.build_tune_text_line(d).decode('ascii').strip()
        self.assertTrue(line.startswith('%TUNE '))
        kvs = dict(p.split('=') for p in line[len('%TUNE '):].split())
        self.assertEqual(int(kvs['max_resumes']), 25)
        self.assertEqual(int(kvs['sd_write_timeout']), 1500)

    def test_all_keys_have_a_binary_command(self):
        # Every key id present in TUNE_KEYS must produce a non-empty
        # binary command — guards against half-added keys.
        for name in th.TUNE_KEYS:
            cmd = th.build_t_command(name, th.TUNE_KEYS[name][2])
            self.assertEqual(cmd[0:1], b'T')
            key_id_in_packet = cmd[1]
            self.assertEqual(key_id_in_packet, th.TUNE_KEYS[name][0])
            # rest = struct-packed value, length matches struct.calcsize
            import struct
            self.assertEqual(
                len(cmd) - 2,
                struct.calcsize(th.TUNE_KEYS[name][1]))


if __name__ == '__main__':
    unittest.main()
