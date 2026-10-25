import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

import pandas as pd

from forex_news_recorder import record_snapshot, evaluate_paper_outcomes


class RecorderTests(unittest.TestCase):
    def state(self):
        return {"events": {}, "history": {"USD|Non-Farm Employment Change": [1,1,1]}, "paper_intents": []}

    def rows(self, actual="", forecast="100"):
        return pd.DataFrame({"event":["Non-Farm Employment Change"], "currency":["USD"],
            "time":[pd.Timestamp("2026-10-09 12:30",tz="UTC")], "time_label":[""],
            "actual":[actual], "forecast":[forecast], "impact":[3]})

    def test_forecast_must_be_observed_before_release_and_entry_is_future(self):
        state=self.state()
        record_snapshot(state,self.rows(),pd.Timestamp("2026-10-09 12:20",tz="UTC"),True)
        observed=pd.Timestamp("2026-10-09 12:35",tz="UTC")
        record_snapshot(state,self.rows(actual="110",forecast="109"),observed,True)
        self.assertEqual(state["events"][next(iter(state["events"]))]["forecast_before_release"],100)
        self.assertTrue(state["paper_intents"])
        for intent in state["paper_intents"]:
            self.assertGreater(pd.Timestamp(intent["planned_entry_at"]),observed)
            self.assertFalse(intent["trade_permission"])
            self.assertIsNone(intent["price"])

    def test_starting_after_release_cannot_create_a_backdated_forecast(self):
        state=self.state()
        record_snapshot(state,self.rows(actual="110"),pd.Timestamp("2026-10-09 12:35",tz="UTC"),True)
        self.assertEqual(state["paper_intents"],[])

    def test_revised_actual_does_not_change_first_value_or_duplicate_intents(self):
        state=self.state()
        record_snapshot(state,self.rows(),pd.Timestamp("2026-10-09 12:20",tz="UTC"),True)
        record_snapshot(state,self.rows(actual="110"),pd.Timestamp("2026-10-09 12:35",tz="UTC"),True)
        first=copy.deepcopy(state)
        record_snapshot(state,self.rows(actual="1"),pd.Timestamp("2026-10-09 12:40",tz="UTC"),True)
        self.assertEqual(state,first)

    def test_intents_do_not_use_a_boundary_before_the_record_can_be_written(self):
        state=self.state()
        record_snapshot(state,self.rows(),pd.Timestamp("2026-10-09 12:20",tz="UTC"),True)
        now=pd.Timestamp("2026-10-09 12:44:59",tz="UTC")
        record_snapshot(state,self.rows(actual="110"),now,True)
        self.assertTrue(all(pd.Timestamp(i["planned_entry_at"])>now+pd.Timedelta(minutes=1)
                            for i in state["paper_intents"]))

    def test_related_releases_in_separate_polls_cancel_before_entry(self):
        state=self.state()
        state["history"]["USD|Unemployment Rate"]=[1,1,1]
        jobs=self.rows()
        unemployment=self.rows()
        unemployment["event"]="Unemployment Rate"
        before=pd.concat([jobs,unemployment],ignore_index=True)
        record_snapshot(state,before,pd.Timestamp("2026-10-09 12:20",tz="UTC"),True)
        record_snapshot(state,self.rows(actual="110"),pd.Timestamp("2026-10-09 12:35",tz="UTC"),True)
        unemployment["actual"]="110"
        record_snapshot(state,unemployment,pd.Timestamp("2026-10-09 12:40",tz="UTC"),True)
        self.assertTrue(state["paper_intents"])
        self.assertTrue(all(i["status"]=="CANCELLED_BEFORE_ENTRY" for i in state["paper_intents"]))

    def test_low_impact_release_is_not_promoted_to_a_high_impact_signal(self):
        state=self.state()
        row=self.rows(actual="110")
        row["impact"]=2
        record_snapshot(state,row,pd.Timestamp("2026-10-09 12:35",tz="UTC"),True)
        self.assertEqual(state["events"],{})

    def test_completed_path_is_paper_only_and_missing_future_prices_are_not_filled(self):
        index=pd.date_range("2026-10-01",periods=650,freq="15min",tz="UTC")
        bars=pd.DataFrame({"Open":1.1,"High":1.1001,"Low":1.0999,"Close":1.1},index=index)
        fills=bars.resample("5min").ffill()
        entry=index[511]+pd.Timedelta(minutes=15)
        def state(when):
            return {"paper_intents":[{"id":"EURUSD|"+when.isoformat(),"symbol":"EURUSD","side":"LONG",
                "planned_entry_at":when.isoformat(),"status":"UNEXECUTED_RESEARCH_INTENT",
                "trade_permission":False,"cost_pips":2.2}]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for tf in ("5m","15m"):
                p=root/f"data/forward_prices/historical_{tf}/EURUSD.csv"
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text("fixture",encoding="utf-8")
            with patch("forex_news_recorder.ROOT",root),patch("forex_news_recorder.load_hourly",side_effect=[bars,fills]):
                completed=evaluate_paper_outcomes(state(entry),entry+pd.Timedelta(hours=5))
            intent=completed["paper_intents"][0]
            self.assertEqual(intent["status"],"SIMULATED_FORWARD_OUTCOME")
            self.assertFalse(intent["trade_permission"])
            self.assertLess(intent["outcome"]["pnl_usd"],0)
            self.assertEqual(len(list((root/"research/news_forward/outcomes").glob("*.csv"))),2)
            future=entry+pd.Timedelta(days=10)
            with patch("forex_news_recorder.ROOT",root),patch("forex_news_recorder.load_hourly",side_effect=[bars,fills]):
                pending=evaluate_paper_outcomes(state(future),future+pd.Timedelta(hours=5))
            self.assertEqual(pending["paper_intents"][0]["status"],"AWAITING_COMPLETE_PRICE_HISTORY")
            self.assertNotIn("outcome",pending["paper_intents"][0])


if __name__ == "__main__":
    unittest.main()
