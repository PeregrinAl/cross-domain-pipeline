from src.data.dataloader import load_config, build_dataloader
from src.utils.seed import set_seed


def main():
    config = load_config("configs/base.yaml")
    set_seed(config["seed"])

    train_loader = build_dataloader(
        config=config,
        split="train",
        domain="source",
        shuffle=True,
    )

    batch = next(iter(train_loader))

    print("Batch keys:", batch.keys())
    print("x_raw shape:", batch["x_raw"].shape)
    print("label shape:", batch["label"].shape)
    print("domains:", batch["domain"])
    print("record_ids:", batch["record_id"])

    if "x_tfr" in batch:
        print("x_tfr shape:", batch["x_tfr"].shape)

    print("Smoke test passed")


if __name__ == "__main__":
    main()