from .unet_att import UnetAttention
import pprint

def build_model(cfg):
    print(f'loading model {cfg.name}')
    if cfg.name == 'unet_att':
        model = UnetAttention()
    else:
        print("Unsupported model type")
        model = None

    model = model.to(cfg.device)       
    return model

class PrettyLog():
    def __init__(self, obj):
        self.obj = obj
    def __repr__(self):
        return pprint.pformat(self.obj)
