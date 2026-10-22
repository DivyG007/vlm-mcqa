"""Small regressions for actual marble classification/truth failure risks."""
import sys,unittest
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/local'))
from probe_iconqa_marbles import objects,answer
from build_iconqa_cf_small import replace


class MarbleTest(unittest.TestCase):
    def test_orange_green_and_purple_are_not_black(self):
        im=Image.new('RGB',(210,50),'white');draw=ImageDraw.Draw(im)
        for k,color in enumerate(['black','orange','green','purple']):
            draw.ellipse((k*50+3,3,k*50+43,43),fill=color)
        obs=objects(im)
        self.assertEqual([o['color'] for o in obs],['black','nonblack','nonblack','nonblack'])
        self.assertEqual(answer(1,len(obs)),'unlikely')

    def test_even_count_boundary_does_not_get_fabricated_label(self):
        with self.assertRaises(ValueError):answer(2,4)

    def test_replacement_changes_truth_with_exact_locality(self):
        im=Image.new('RGB',(200,50),'white');d=ImageDraw.Draw(im)
        for k in range(4):d.ellipse((k*50+3,3,k*50+43,43),fill='black' if k==0 else 'blue')
        obs=objects(im);sprite=im.crop(obs[1]['bbox'])
        cf,mask,info=replace(im,obs,0,sprite,'impossible')
        self.assertEqual(info['black_after'],0)
        a=np.array(im);b=np.array(cf);box=obs[0]['bbox'];a[box[1]:box[3],box[0]:box[2]]=b[box[1]:box[3],box[0]:box[2]]
        self.assertTrue(np.array_equal(a,b))
        self.assertGreater(np.array(mask).sum(),0)


if __name__=='__main__':unittest.main()
