# ============================================================
# STEP 2 — RUNTIME & GPU CHECK
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("RUNTIME CHECK")
print("=" * 60)
print("Device :", device)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print("GPU    :", gpu_name)
    print(f"VRAM   : {total_vram:.2f} GB")
    print("CUDA   :", torch.version.cuda)

    x = torch.randn(2, 3, 64, 64, device=device)
    print("CUDA tensor test :", x.device)
    del x
    torch.cuda.empty_cache()

    print("\nGPU CHECK PASSED")
else:
    print("\nWARNING: CUDA GPU NOT AVAILABLE")
    print("Change Colab runtime to GPU before training.")
