"""Ensure reduced sampling retains independent sources and valid donors."""
import json,random,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/ada'))
from iconqa_small_inputs import select_pairs,verify_release
from vlm_mcqa.clevr_mcq4.variants import build_variants,_contrast,build_random_pair_contrasts


def fixture(n=33):
    variants={};contrasts=[];pv={}
    for k in range(n):
        pair=f'fixture_{k:02}';members={}
        for member,answer in [('x','certain'),('y','impossible')]:
            item={'item_id':pair+'_'+member,'split':'discovery','scene_id':pair,'counterfactual_pair_id':pair,
                  'pair_member':member,'image':pair+'_'+member+'.png','question':'Probability?',
                  'question_family':'probability','program_depth':None,
                  'option_contents':['certain','impossible','probable','unlikely'],'semantic_answer':answer}
            vs=build_variants(item,permuted_control=True);variants.update({v['variant_id']:v for v in vs})
            members[member]={v['correct_index']:v for v in vs if v['label_scheme']=='letters' and v['variant_kind']=='position'}
            members[member+'p']={v['correct_index']:v for v in vs if v['variant_kind']=='distractor_permuted'}
        x,y=members['x'],members['y']
        def reciprocal(kind,a,b):
            for donor,recipient in ((a,b),(b,a)):
                extra={}
                if kind=='content_and_position':
                    p=recipient['option_contents'].index(donor['semantic_answer'])
                    extra={'content_index':p,'content_symbol':recipient['labels'][p]}
                contrasts.append(_contrast(kind,donor,recipient,**extra))
        reciprocal('position',x[0],x[1]);reciprocal('position',y[2],y[3])
        reciprocal('content',x[0],y[0]);reciprocal('content',x[2],y[2])
        reciprocal('content_and_position',x[0],y[2]);reciprocal('content_and_position',x[1],y[3])
        reciprocal('identity',members['xp'][0],x[0]);reciprocal('identity',members['yp'][2],y[2])
        pv[pair]=[v for v in variants.values() if v['counterfactual_pair_id']==pair]
    contrasts+=build_random_pair_contrasts(pv,random.Random(1))
    session=SimpleNamespace(variants=variants,clean=lambda v:SimpleNamespace(correct=not v.startswith('fixture_00_'),readout={'margin':1.}))
    return session,contrasts


class SmallInputsTest(unittest.TestCase):
    def test_32_source_units_and_closed_donors(self):
        s,cs=fixture();reduced,coverage,head,recipients=select_pairs(s,'discovery',cs)
        self.assertEqual(coverage['pairs_used'],32)
        self.assertNotIn('fixture_00',coverage['pair_ids'])
        self.assertEqual(len(reduced),576);self.assertEqual(len(head),64)
        self.assertEqual(len({s.variants[v]['counterfactual_pair_id'] for v in recipients}),32)
        self.assertEqual(Counter('native' if '_x|' in v else 'edited' for v in recipients),{'native':16,'edited':16})
        selected=set(coverage['pair_ids'])
        self.assertTrue(all(s.variants[c['donor_variant_id']]['counterfactual_pair_id'] in selected for c in reduced))

    def test_underfill_is_explicit(self):
        s,cs=fixture(4);reduced,coverage,head,recipients=select_pairs(s,'discovery',cs,target=4)
        self.assertEqual(coverage['pairs_used'],3);self.assertEqual(reduced,[])

    def test_unapproved_release_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'release_manifest.json').write_text(json.dumps({'files':{},'release_sha256':'wrong'}))
            with self.assertRaisesRegex(ValueError,'binding changed'):verify_release(root)


if __name__=='__main__':unittest.main()
