"""Tests for sd_health.py — runs without a real SD card by using tmpdir."""
import datetime
import os
import sqlite3
import sys
import tempfile
import unittest

# import the module under test
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import sd_health  # noqa: E402


class TestLiveWriteTest(unittest.TestCase):
    def test_happy_path_clean(self):
        with tempfile.TemporaryDirectory() as td:
            res = sd_health.live_write_test(td, payload_mb=1, chunk_kb=8)
        self.assertTrue(res['ok'], msg=res)
        self.assertEqual(res['bytes_written'], 1024 * 1024)
        self.assertEqual(res['write_errors'], 0)
        self.assertEqual(res['verify_errors'], 0)
        # tmpfs / disk is wildly faster than DEGRADING_LATENCY_P95_MS;
        # a non-zero but tiny p95 is the expected shape here.
        self.assertGreaterEqual(res['latency_p95_ms'], 0.0)
        self.assertLess(res['latency_p95_ms'], 1000.0)

    def test_bad_dir(self):
        res = sd_health.live_write_test('/nonexistent-path-xyz', payload_mb=1)
        self.assertFalse(res['ok'])
        self.assertIsNotNone(res['error'])

    def test_bad_size(self):
        with tempfile.TemporaryDirectory() as td:
            # chunk > payload — bail out before doing any I/O
            res = sd_health.live_write_test(td, payload_mb=1, chunk_kb=4096)
        self.assertFalse(res['ok'])
        self.assertIn('bad payload/chunk size', res['error'])

    def test_probe_cleaned_up(self):
        with tempfile.TemporaryDirectory() as td:
            res = sd_health.live_write_test(td, payload_mb=1)
            self.assertFalse(os.path.exists(res['probe_path']),
                             f'probe left behind: {res["probe_path"]}')

    def test_keep_probe(self):
        with tempfile.TemporaryDirectory() as td:
            res = sd_health.live_write_test(td, payload_mb=1, keep_probe=True)
            self.assertTrue(os.path.exists(res['probe_path']))


class TestParseCkptSummary(unittest.TestCase):
    def _write_txt(self, lines, td):
        p = os.path.join(td, 'sample.TXT')
        with open(p, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return p

    def test_no_ckpts(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write_txt([
                '%STOP AT',
                '00,000000,000000,000000,000000,000000,000000,000000,000000,FD00,FAD0,1EB0',
            ], td)
            res = sd_health.parse_ckpt_summary(p)
        self.assertEqual(res['ckpt_count'], 0)
        self.assertEqual(res['ckpt_errs'], 0)
        self.assertEqual(res['ckpt_extretries'], 0)
        self.assertEqual(res['duration_s'], 0.0)

    def test_two_ckpts(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write_txt([
                '%META {}',
                '%CKPT t=60000 b=120 e=0 r=0 n=0 o=0 x=0',
                '00,000000,000000,000000,000000,000000,000000,000000,000000',
                '%CKPT t=120000 b=240 e=2 r=3 n=0 o=0 x=1',
                '%Total time 120s',
            ], td)
            res = sd_health.parse_ckpt_summary(p)
        self.assertEqual(res['ckpt_count'], 2)
        self.assertEqual(res['ckpt_errs'], 2)
        self.assertEqual(res['ckpt_retries'], 3)
        self.assertEqual(res['ckpt_extretries'], 1)
        self.assertEqual(res['duration_s'], 60.0)

    def test_malformed_ckpt_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._write_txt([
                '%CKPT garbage line with no kv',
                '%CKPT t=60000 b=120 e=5 r=0 n=0 o=0 x=0',
            ], td)
            res = sd_health.parse_ckpt_summary(p)
        self.assertEqual(res['ckpt_count'], 1)
        self.assertEqual(res['ckpt_errs'], 5)

    def test_unknown_keys_ignored(self):
        # Forward-compat: future firmware adds e.g. T=<hash> for the
        # tunable summary; parser must skip silently.
        with tempfile.TemporaryDirectory() as td:
            p = self._write_txt([
                '%CKPT t=60000 b=120 e=0 r=0 n=0 o=0 x=0 T=abc123',
            ], td)
            res = sd_health.parse_ckpt_summary(p)
        self.assertEqual(res['ckpt_count'], 1)


class TestHealthVerdict(unittest.TestCase):
    def _live(self, **kw):
        base = dict(
            ok=True, bytes_written=1024 * 1024, wall_time_s=0.5,
            mb_per_s=2.0,
            latency_p50_ms=10.0, latency_p95_ms=50.0, latency_max_ms=80.0,
            write_errors=0, verify_errors=0, error=None, probe_path='')
        base.update(kw)
        return base

    def _ckpt(self, **kw):
        base = dict(
            ckpt_count=720, first_t_ms=0, last_t_ms=43_200_000,
            duration_s=43200.0,
            ckpt_errs=0, ckpt_retries=0, ckpt_reinits=0,
            ckpt_overruns=0, ckpt_extretries=0)
        base.update(kw)
        return base

    def test_healthy(self):
        v, n = sd_health.health_verdict(self._live(), self._ckpt())
        self.assertEqual(v, 'HEALTHY')
        self.assertEqual(n, 'clean')

    def test_degrading_latency(self):
        v, n = sd_health.health_verdict(
            self._live(latency_p95_ms=400.0), self._ckpt())
        self.assertEqual(v, 'DEGRADING')
        self.assertIn('latency_p95', n)

    def test_dying_latency(self):
        v, n = sd_health.health_verdict(
            self._live(latency_p95_ms=1500.0), self._ckpt())
        self.assertEqual(v, 'DYING')

    def test_dying_verify(self):
        v, _ = sd_health.health_verdict(
            self._live(verify_errors=1), self._ckpt())
        self.assertEqual(v, 'DYING')

    def test_dying_write_error(self):
        v, _ = sd_health.health_verdict(
            self._live(write_errors=1, ok=False), self._ckpt())
        self.assertEqual(v, 'DYING')

    def test_degrading_ckpt(self):
        v, _ = sd_health.health_verdict(
            self._live(), self._ckpt(ckpt_extretries=10))
        self.assertEqual(v, 'DEGRADING')

    def test_dying_ckpt(self):
        v, _ = sd_health.health_verdict(
            self._live(), self._ckpt(ckpt_extretries=100))
        self.assertEqual(v, 'DYING')

    def test_reinit_is_degrading(self):
        # any reinit fired = always-worth-noting, even with otherwise
        # clean counters.
        v, n = sd_health.health_verdict(
            self._live(), self._ckpt(ckpt_reinits=1))
        self.assertEqual(v, 'DEGRADING')
        self.assertIn('reinits', n)

    def test_dying_beats_degrading(self):
        # latency is DEGRADING but verify is DYING — final = DYING.
        v, _ = sd_health.health_verdict(
            self._live(latency_p95_ms=400.0, verify_errors=2),
            self._ckpt(ckpt_extretries=10))
        self.assertEqual(v, 'DYING')


class TestPersistVerdict(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, 'sessions.db')
            live = dict(ok=True, mb_per_s=2.0, latency_p50_ms=10.0,
                        latency_p95_ms=50.0, latency_max_ms=80.0,
                        write_errors=0, verify_errors=0,
                        bytes_written=1024 * 1024, wall_time_s=0.5,
                        error=None, probe_path='')
            ckpt = dict(ckpt_count=720, ckpt_errs=2, ckpt_retries=3,
                        ckpt_reinits=0, ckpt_overruns=0, ckpt_extretries=1,
                        duration_s=43200.0)
            dts = datetime.datetime(2026, 5, 14, 8, 0, 0)
            ok = sd_health.persist_verdict(
                db, dts, '/mnt/sd', 'OBCI_01.TXT', live, ckpt,
                'HEALTHY', 'clean')
            self.assertTrue(ok)
            # Read back
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute('SELECT verdict, ckpt_errs, mb_per_s FROM SdHealth')
            rows = cur.fetchall()
            con.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], 'HEALTHY')
            self.assertEqual(rows[0][1], 2)
            self.assertAlmostEqual(rows[0][2], 2.0)

    def test_replace_on_duplicate_dts(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, 'sessions.db')
            live = dict(ok=True, mb_per_s=2.0, latency_p50_ms=10.0,
                        latency_p95_ms=50.0, latency_max_ms=80.0,
                        write_errors=0, verify_errors=0,
                        bytes_written=1024 * 1024, wall_time_s=0.5,
                        error=None, probe_path='')
            ckpt = dict(ckpt_count=0, ckpt_errs=0, ckpt_retries=0,
                        ckpt_reinits=0, ckpt_overruns=0, ckpt_extretries=0,
                        duration_s=0.0)
            dts = datetime.datetime(2026, 5, 14, 8, 0, 0)
            sd_health.persist_verdict(db, dts, '/mnt/sd', 'a.TXT', live,
                                      ckpt, 'HEALTHY', 'clean')
            sd_health.persist_verdict(db, dts, '/mnt/sd', 'a.TXT', live,
                                      ckpt, 'DYING', 'flaky')
            con = sqlite3.connect(db)
            n = con.execute('SELECT COUNT(*) FROM SdHealth').fetchone()[0]
            verdict = con.execute('SELECT verdict FROM SdHealth').fetchone()[0]
            con.close()
            self.assertEqual(n, 1)
            self.assertEqual(verdict, 'DYING')


class TestRunHealthCheck(unittest.TestCase):
    def test_end_to_end_no_txt(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, 'sessions.db')
            v, n, live, ckpt, report = sd_health.run_health_check(
                td, txt_file=None, session_db=db)
            self.assertIn(v, ('HEALTHY', 'DEGRADING', 'DYING'))
            self.assertIsInstance(report, str)
            self.assertIn('SD Health', report)
            # DB row got written
            con = sqlite3.connect(db)
            n_rows = con.execute('SELECT COUNT(*) FROM SdHealth').fetchone()[0]
            con.close()
            self.assertEqual(n_rows, 1)

    def test_end_to_end_with_txt(self):
        with tempfile.TemporaryDirectory() as td:
            txt = os.path.join(td, 'OBCI_01.TXT')
            with open(txt, 'w') as f:
                f.write('%META {}\n')
                f.write('%CKPT t=60000 b=120 e=0 r=0 n=0 o=0 x=0\n')
                f.write('00,000000,000000,000000,000000,000000,000000,000000,000000\n')
                f.write('%CKPT t=120000 b=240 e=0 r=0 n=0 o=0 x=0\n')
            db = os.path.join(td, 'sessions.db')
            v, n, live, ckpt, report = sd_health.run_health_check(
                td, txt_file=txt, session_db=db)
            self.assertEqual(ckpt['ckpt_count'], 2)
            self.assertEqual(ckpt['ckpt_errs'], 0)


if __name__ == '__main__':
    unittest.main()
