#########################################################################################################
#
#   ELEC 475 - Lab 1 (Steps 4, 5, 6)
#   Erhowvosere Otubu - 20293052
#   Mihran Asadullah - 20285090
#   Fall 2025
#

import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer

def main():
    parser = argparse.ArgumentParser()
    # -l: the filename of the train model weights
    parser.add_argument('-l', type=str, required=True, help='weights file, e.g. MLP.8.pth')
    # -z: bottleneck size used when training
    parser.add_argument('-z', type=int, default=8, help='bottleneck size [default: 8]')
    # -n: number of interpolation steps
    parser.add_argument('-n', type=int, default=8, help='interpolation steps (excl. endpoints) [8]')
    #: -r: number of rows to draw
    parser.add_argument('-r', type=int, default=3, help='number of interpolation rows [3]')
    args = parser.parse_args()

    # Device selection between GPU or CPU
    # Program selects GPU if CUDA is available, otherwise it selects the CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # Dataset
    tfm = transforms.ToTensor()
    train_set = MNIST('./data/mnist', train=True, download=True, transform=tfm)

    # Build model and load weights
    N = 28 * 28
    model = autoencoderMLP4Layer(N_input=N, N_bottleneck=args.z, N_output=N)
    model.load_state_dict(torch.load(args.l, map_location=device))
    model.to(device)
    model.eval()

    #4: Test Your Autoencoder
    while True:
        s = input("Step 4 — Enter index [0..59999], or blank to continue: ").strip()
        if s == "": break
        idx = int(s)

        img_u8 = train_set.data[idx]
        img = img_u8.to(torch.float32) / 255.0
        x = img.view(1, -1).to(device)

        with torch.no_grad():
            out = model(x).view(28, 28).cpu()

        f = plt.figure()
        f.add_subplot(1,2,1); plt.imshow(img_u8, cmap='gray'); plt.title('Input')
        f.add_subplot(1,2,2); plt.imshow(out,    cmap='gray'); plt.title('Reconstruction')
        plt.tight_layout(); plt.show()

    #5: Image Denoising
    while True:
        s = input("Step 5 — Enter index for denoising [0..59999], or blank to continue: ").strip()
        if s == "": break
        idx = int(s)

        img = train_set.data[idx].type(torch.float32)
        img = (img - torch.min(img)) / torch.max(img)  # original normalize
        img_noise = img + torch.rand(img.shape)         # original noise

        x_noisy = img_noise.to(device).view(1, -1).type(torch.FloatTensor)
        with torch.no_grad():
            out = model(x_noisy).view(28, 28).type(torch.FloatTensor)

        f = plt.figure()
        f.add_subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title('Input')
        f.add_subplot(1,3,2); plt.imshow(img_noise, cmap='gray'); plt.title('Noise')
        f.add_subplot(1,3,3); plt.imshow(out, cmap='gray'); plt.title('Denoised Output')
        plt.tight_layout(); plt.show()

    #6: Bottleneck Interpolation
    print(f"Step 6 — Enter {args.r} pairs like: 12 345   (0..{train_set.data.size(0)-1})")
    pairs = []
    for r in range(args.r):
        s = input(f"  Row {r+1} indices > ").strip()
        if s == "": break
        a, b = s.split()
        pairs.append((int(a), int(b)))

    if pairs:
        cols = args.n + 2  # start + mids + end
        fig = plt.figure(figsize=(1.6*cols, 2.2*len(pairs)))
        with torch.no_grad():
            for row, (i1, i2) in enumerate(pairs):
                # flatten helper inline to keep it simple
                x1 = (train_set.data[i1].to(torch.float32) / 255.0).view(1, -1).to(device)
                x2 = (train_set.data[i2].to(torch.float32) / 255.0).view(1, -1).to(device)

                z1 = model.encode(x1)
                z2 = model.encode(x2)

                ts = torch.linspace(0.0, 1.0, steps=args.n + 2, device=device)[1:-1]

                # start
                ax = fig.add_subplot(len(pairs), cols, row*cols + 1)
                ax.imshow(model.decode(z1).view(28,28).cpu(), cmap='gray')

                # intermediates
                for k, t in enumerate(ts, start=2):
                    zt = (1.0 - t) * z1 + t * z2
                    ax = fig.add_subplot(len(pairs), cols, row*cols + k)
                    ax.imshow(model.decode(zt).view(28,28).cpu(), cmap='gray')

                # end
                ax = fig.add_subplot(len(pairs), cols, row*cols + cols)
                ax.imshow(model.decode(z2).view(28,28).cpu(), cmap='gray')

        plt.tight_layout(); plt.show()
    else:
        print("No pairs provided for interpolation.")

if __name__ == "__main__":
    main()
