import unittest

import numpy as np
import pandas as pd

from research_news_surprise import pair_states, surprise_signals


class NewsSurpriseTests(unittest.TestCase):
    def events(self):
        return pd.DataFrame({"id": range(8), "series": [1]*8,
            "name": ["Non-Farm Employment Change"]*8, "currency": ["USD"]*8,
            "impact": ["high"]*8, "actual": [101,99,101,99,103,97,101,101],
            "forecast": [100]*8, "time": pd.date_range("2025-01-01 13:30", periods=8, freq="MS", tz="UTC")})

    def test_future_releases_do_not_change_past_scale_or_votes(self):
        data = self.events()
        original = surprise_signals(data)
        changed = data.copy()
        changed.loc[6:,"actual"] = 10000
        altered = surprise_signals(changed)
        pd.testing.assert_frame_equal(original[original.id<6], altered[altered.id<6])

    def test_signal_waits_for_dissemination_and_closed_bar(self):
        signals = surprise_signals(self.events())
        self.assertTrue((signals.decision_at >= signals.time + pd.Timedelta(minutes=5)).all())
        self.assertTrue((signals.decision_at.dt.minute % 15 == 0).all())

    def test_currency_strength_maps_to_correct_pair_direction_and_conflicts_cancel(self):
        time = pd.Timestamp("2025-01-01 14:00",tz="UTC")
        signals = pd.DataFrame({"currency":["USD"],"vote":[1],"decision_at":[time]})
        index = pd.date_range(time-pd.Timedelta(hours=1),periods=8,freq="15min")
        eur, _ = pair_states(signals,"EURUSD",index)
        jpy, _ = pair_states(signals,"USDJPY",index)
        np.testing.assert_array_equal(eur,-jpy)
        self.assertEqual(eur[3],-1)
        conflict=pd.concat([signals,pd.DataFrame({"currency":["EUR"],"vote":[1],"decision_at":[time]})])
        states, positions=pair_states(conflict,"EURUSD",index)
        self.assertEqual(len(positions),0)

    def test_unknown_news_and_equal_actual_consensus_are_excluded(self):
        data = self.events()
        data.loc[4,"name"]="CPI y/y"
        data.loc[5,"actual"]=100
        signals=surprise_signals(data)
        self.assertNotIn(4,signals.id.tolist())
        self.assertNotIn(5,signals.id.tolist())


if __name__ == "__main__":
    unittest.main()
