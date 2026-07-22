# ==========================================================
# STEP 18 : Image Transformation
# ==========================================================

from torchvision import transforms

transform = transforms.Compose([

    transforms.Resize(
        (CONFIG["image_size"], CONFIG["image_size"])
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])

print("="*60)
print("Image Transform Ready")
print("="*60)

print(transform)
