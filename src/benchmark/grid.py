from itertools import product


def build_stage_1_grid(config: dict) -> list[dict]:
    stage_cfg = config["benchmark"]["stage_1_screening"]
    datasets = [d for d in config["datasets"] if d.get("enabled", True)]

    rows = []
    for dataset, preprocessing, representation, adaptation in product(
        datasets,
        stage_cfg["preprocessing"],
        stage_cfg["representations"],
        stage_cfg["adaptation"],
    ):
        rows.append(
            {
                "stage": "stage_1_screening",
                "dataset_id": dataset["dataset_id"],
                "preprocessing": preprocessing,
                "representation": representation,
                "adaptation": adaptation,
            }
        )
    return rows


def build_stage_2_grid(config: dict, selected_pairs: list[dict]) -> list[dict]:
    stage_cfg = config["benchmark"]["stage_2_adaptation"]

    rows = []
    for pair, adaptation in product(selected_pairs, stage_cfg["adaptation"]):
        rows.append(
            {
                "stage": "stage_2_adaptation",
                "dataset_id": pair["dataset_id"],
                "preprocessing": pair["preprocessing"],
                "representation": pair["representation"],
                "adaptation": adaptation,
            }
        )
    return rows