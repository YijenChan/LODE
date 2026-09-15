import unittest
from lode.commit import commit_case

class CommitTests(unittest.TestCase):
    def fixture(self):
        return [dict(id=i,src=a,dst=b,rel='EVENT_WRITE',ts=i) for i,(a,b) in enumerate([(0,1),(1,2),(2,3),(1,8),(8,9)])]
    def test_no_claim_no_commit(self):
        events=self.fixture();result,_=commit_case(events,[],0);self.assertFalse(result)
    def test_preserves_connectors_prunes_side_branch(self):
        events=self.fixture();result,_=commit_case(events,[3],0)
        self.assertEqual([e['id'] for e in result],[0,1,2]);self.assertEqual(result,events[:3])
    def test_never_invents_graph_or_direct_labels(self):
        events=self.fixture();direct=[3];result,_=commit_case(events,direct,0)
        self.assertEqual(direct,[3]);self.assertTrue(all(e in events for e in result))

if __name__=='__main__':unittest.main()
