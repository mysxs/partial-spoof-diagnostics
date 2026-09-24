import unittest
import numpy as np
from event_strata import spans,match_events,duration_bin,boundary_stratum


class EventTests(unittest.TestCase):
    def test_spans_retains_edges_and_empty_genuine(self):
        np.testing.assert_allclose(spans([1,1,0,1],.01),[[0,.02],[.03,.04]])
        self.assertEqual(spans([0,0],.01).shape,(0,2))

    def test_split_prediction_cannot_count_twice(self):
        matches=match_events([[0,1]],[[0,.5],[.5,1]])
        self.assertEqual(len(matches),1)
        self.assertEqual(match_events([[0,1]],[[2,3]]),[])

    def test_duration_and_boundary_rounding(self):
        self.assertEqual(duration_bin(.3-.2),'100to300')
        self.assertEqual(duration_bin(1),'600to1000')
        self.assertEqual(boundary_stratum([.1,.2],[0,.08,.22,.3]),'both_near')
        self.assertEqual(boundary_stratum([.1,.2],[0,.3],True),'alignment_fallback')


if __name__=='__main__':unittest.main()
