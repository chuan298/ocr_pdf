import argparse

from model.trainer import Trainer
from tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/home/ancv/Work/ocr_pdf/vietocr/config/vgg-transformer.yml", help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
    args = parser.parse_args()
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
    config = Cfg.load_config_from_file(args.config)
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
    print(args)
    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    trainer.train()

if __name__ == '__main__':
    main()
