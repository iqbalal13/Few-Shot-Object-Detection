
import torch
import json
import gc

for shot in (1, 3, 5):
    folder = WORKSPACE / "outputs" / f"stage2_cctv_{shot}shot"

    history = json.loads(
        (folder / "history.json").read_text()
    )

    ckpt = torch.load(
        folder / "best.pth",
        map_location="cpu",
        weights_only=False
    )

    best_record = max(
        history,
        key=lambda x: x["val_AP50_95"]
    )

    selection = ckpt["selection"]

    assert len(history) == 10
    assert ckpt["shot_images"] == shot
    assert selection["best_epoch"] == best_record["epoch"]
    assert abs(
        selection["score"] - best_record["val_AP50_95"]
    ) < 1e-6

    print(
        f"{shot}-shot VALID | "
        f"Best epoch: {best_record['epoch'] + 1} | "
        f"AP: {selection['score']:.5f}"
    )

    del ckpt
    gc.collect()

print("ALL CHECKPOINTS VERIFIED")
