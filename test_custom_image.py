import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from train_unet_div2k import UNet256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet256().to(device)
model.load_state_dict(torch.load("unet_div2k_256.pth"))
model.eval()

def test_hd(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    orig = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    recon = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig)
    axes[0].set_title("Input (256x256 HD)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(recon)
    axes[1].set_title("Reconstructed (U-Net DIV2K)", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

test_hd("test.jpg")