"""Canonical model label order."""

PATHOLOGY_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Support Devices",
)


def validate_labels(labels: list[str] | tuple[str, ...]) -> None:
    if tuple(labels) != PATHOLOGY_LABELS:
        raise ValueError(
            "Dataset labels do not match the canonical 17-label order. "
            f"Expected {PATHOLOGY_LABELS}, received {tuple(labels)}."
        )
