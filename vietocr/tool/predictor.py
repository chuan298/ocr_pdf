from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict, batch_process_image, batch_translate_beam_search
from vietocr.tool.utils import download_weights

import torch

class Predictor():
    def __init__(self, config):

        device = config['device']
        
        model, vocab = build_model(config)
        weights = '/tmp/weights.pth'

        if config['weights'].startswith('http'):
            weights = download_weights(config['weights'])
        else:
            weights = config['weights']

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        
    def predict(self, img, return_prob=False):
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = translate(img, self.model)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)
        
        if return_prob:
            return s, prob
        else:
            return s
    
    def batch_predict(self, imgs, files, return_prob=False):
        
        list_batch_imgs, list_batch_files = batch_process_image(imgs, files, self.config['dataset']['image_height'], self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'], self.config['dataset']['batch_size'])        
        
        list_batch_imgs = [img.to(self.config['device'], non_blocking=True) for img in list_batch_imgs]

        result_s, result_probs = [], []
        if self.config['predictor']['beamsearch']:
            for batch_imgs in list_batch_imgs:
                sent = translate_beam_search(batch_imgs, self.model)
                for w in sent:
                    result_s.append(self.vocab.decode(w.tolist()))
            result_probs = None
        else:
            for batch_imgs in list_batch_imgs:
                s, prob = translate(batch_imgs, self.model)
                # print(s.shape, prob.shape)
                for w in s:
                    result_s.append(self.vocab.decode(w.tolist()))
                result_probs = prob
        if return_prob:
            return result_s, result_probs, list_batch_files
        else:
            return result_s, list_batch_files
