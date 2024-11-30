import os
from model import CMTFusion
import torch
from torchvision.utils import save_image
import utils
import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='./result_images/KAIST/', help='path of fused image')
    parser.add_argument('--dataset', type=str, default='./dataset_inference/KAIST/', help='path of source image')
    parser.add_argument('--dataset_name', type=str, default='KAIST', help='dataset name')
    args = parser.parse_args()

    if os.path.exists(args.out_path) is False:
        os.mkdir(args.out_path)

    # device setting for gpu users
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    fusion_model = torch.nn.DataParallel(CMTFusion(), device_ids=[0])
    fusion_model.load_state_dict(
        torch.load("./models/pretrained.pth", map_location=torch.device('cpu')))
    print("===>Testing using dataset: ", args.dataset_name)
    fusion_model.to(device)
    fusion_model.eval()

    with torch.no_grad():
        for i in range(len(os.listdir(args.dataset))):
            index = i + 1
            infrared_path = args.dataset + 'IR' + str(index) + '.png'
            visible_path = args.dataset + 'VIS' + str(index) + '.png'
            print(index)
            if os.path.isfile(infrared_path):
                real_ir_imgs = utils.get_test_images(infrared_path, height=None, width=None)
                real_rgb_imgs = utils.get_test_images(visible_path, height=None, width=None)

                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                    
                fused, fused2, fused3 = fusion_model(real_rgb_imgs.to(device), real_ir_imgs.to(device))
                # # save images
                save_image(fused, os.path.join(args.out_path, "%d.png" % index), normalize=True)

    print('Done......')


if __name__ == '__main__':
    run()