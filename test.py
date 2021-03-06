import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import sys

# philipedwards change begin
# Work out the root directory of ESRGAN
# this allows it to be executed from outside this directory.
ESRGANPath = osp.realpath(__file__)
ESRGANPath = ESRGANPath[:-len(osp.basename(ESRGANPath))]

# models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
modelfile = 'RRDB_ESRGAN_x4.pth'
# Instructions say that this already existed, but I couldn't find it.
# This allows you to specify the model as the first positional argument
if len(sys.argv) > 1:
    print("model specified: " + sys.argv[1])
    modelfile = sys.argv[1]

model_path = osp.join(ESRGANPath, 'models', modelfile)
# philipedwards change end

device = torch.device('cpu')
if torch.cuda.device_count() > 0:
    print("Using CUDA device")
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
else:
    print("No CUDA device found, using CPU fallback")

img_src_path = osp.join(ESRGANPath, 'LR/*')
img_dst_path = osp.join(ESRGANPath, "results")
print(img_src_path)

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(img_src_path):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(img_dst_path + '/{:s}_rlt.png'.format(base), output)

print("Finished ESRGAN")
