import argparse, os
from PIL import Image

from tool.predictor import Predictor
from tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default="/home/ancv/Work/KhmOcr/real_data_test/crop_box/", help='foo help')
    parser.add_argument('--config', default="/home/ancv/Work/OCR/vietocr/config/vgg-transformer.yml", help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)

    # img = Image.open(args.img)

    list_files = os.listdir(args.img)

    list_imgs = []
    for file in list_files:
        img = Image.open(args.img + file)
        list_imgs.append(img)

    result_s, list_batch_files = detector.batch_predict(list_imgs, list_files)

    for i, s in enumerate(result_s):
        # index = list_files.index(list_batch_files[i])
        
        with open("predict_latin_transfomer.log", "a", encoding="utf-8") as f:
            for j in "<":
                if j in result_s[i]: 
                    f.write(list_batch_files[i] + " || " + result_s[i]  + "\n")
                    break

if __name__ == '__main__':
    main()
