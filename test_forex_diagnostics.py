import unittest

from forex_diagnostics import engine_evidence_summary, funnel_rows


class EvidenceSummaryTests(unittest.TestCase):
    def test_transition_does_not_claim_test_missing(self):
        for qualities in ({}, {"RANGE": {"trade_count": 2}}):
            result = engine_evidence_summary(None, qualities)
            self.assertEqual(result["label"], "Piyasa kararsız")

    def test_range_uses_its_own_count(self):
        result = engine_evidence_summary("RANGE", {
            "TREND": {"trade_count": 1},
            "RANGE": {"trade_count": 2, "required_trades": 60},
        })
        self.assertEqual(result["label"], "Örnek yetersiz")
        self.assertIn("2 işlem", result["text"])
        self.assertNotIn("1 işlem", result["text"])

    def test_zero_trades_is_completed_but_insufficient(self):
        self.assertEqual(engine_evidence_summary("RANGE", {
            "RANGE": {"trade_count": 0},
        })["label"], "Örnek yetersiz")
        self.assertEqual(engine_evidence_summary("RANGE", {})["label"], "Test yapılmalı")

    def test_approval_needs_both_quality_and_evidence(self):
        for label, edge, expected in [
            ("İyi", "DOĞRULANDI", "Onaylandı"),
            ("Zayıf", "DOĞRULANDI", "Onaylanmadı"),
            ("Orta", "ADAY / DEMO", "Onaylanmadı"),
        ]:
            with self.subTest(label=label, edge=edge):
                self.assertEqual(engine_evidence_summary("TREND", {
                    "TREND": {"label": label, "edge": {"label": edge}, "trade_count": 80},
                })["label"], expected)

    def test_funnel_counts_sequential_rejections(self):
        rows = funnel_rows({"İncelenen": 100, "Tetik": 10, "Seans": 4, "İşlem": 2})
        self.assertEqual([r["Bu aşamada elenen"] for r in rows], [0, 90, 6, 2])
        self.assertEqual(funnel_rows({}), [])


if __name__ == "__main__":
    unittest.main()
