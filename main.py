from tools.train import Trainer
from tools.test import Tester
import yaml
from easydict import EasyDict as edict

if __name__=='__main__':
    with open('configs/unet_att_mps.yaml') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
        cfgs = edict(cfgs)

    if cfgs.model.train == True:
        trainer = Trainer(cfgs)
        trainer.train()

    if cfgs.model.test == True:
        tester = Tester(cfgs)
        tester.test()
    
    print()