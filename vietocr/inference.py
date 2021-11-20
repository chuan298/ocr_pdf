import argparse, os
from PIL import Image

from tool.predictor import Predictor
from tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_test', default="/var/account/ancv/labels_test.txt", help='foo help')
    parser.add_argument('--config', default="/var/account/ancv/vietocr/config/vgg-transformer.yml", help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)

    list_files = []
    list_labels = []
    with open(args.file_test, "r", encoding="utf-8") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            # count += 1
            # if count > 10: break
            line = line.strip().split("\t")
            list_files.append(line[0])
            list_labels.append(line[1])


    list_imgs = []
    for file in list_files:
        img = Image.open("/var/account/ancv/" + file)
        list_imgs.append(img)
    
    result_s, list_batch_files = detector.batch_predict(list_imgs, list_files)
    # print(list_batch_files)
    num_true = 0
    for i, s in enumerate(result_s):
        index = list_files.index(list_batch_files[i])
        # print(result_s[i] + " || " + list_labels[index])
        if result_s[i] == list_labels[index]: num_true += 1
        else:
            with open("inferrence_error.log", "a", encoding="utf-8") as f:
                f.write(result_s[i] + " || " + list_labels[index]  + "\n")
    print(num_true / len(result_s))

if __name__ == '__main__':
    main()
