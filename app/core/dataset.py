import shutil
import random
import yaml
import json
from pathlib import Path
from typing import Iterable, Any
from datetime import datetime


# --- Create a YOLO-ready dataset ---
def _create_yolo_dataset(
        source_path: str | Path, # path where we have our "images/" & "labels/" pair
        dest_path: str | Path = None, # if none: dataset will be created inside the source directory.
        val_frac: float = 0.2,
        shuffle: bool = True,
        seed: int | None = None,
) -> None:
    """
    Use this function to create a YOLO training dataset. We need to provide a path where 'images/' and 'labels/' sit.
    Note that for a less strained process, it is highly recommended to include 'classes.txt' alongside these folders.
    ---
    THIS FUNCTION IS SENSITIVE TO LABEL CLASSES, USE WITH DISCRETION
    :param source_path: Folder containing paired 'images/' and 'labels/'
    :param dest_path: YOLO Dataset
    :param val_frac: What fraction of the dataset do we allocate to a YOLO validation set?
    :param shuffle: Shuffle image order (usually want True)
    :param seed: Shuffle seed
    :return: None
    """

    # guard against val frac values
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")

    source_path = Path(source_path)
    dataset_name = source_path.stem + "_dataset"

    # resolve destination path
    if dest_path is None:
        dest_path = source_path / dataset_name
    else:
        dest_path = Path(dest_path)

    # check that the source exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source '{source_path}' does not exist")

    images_dir = source_path / "images"
    labels_dir = source_path / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Source '{source_path}' must contain 'images/' and 'labels/' directories")

    # create destination
    dest_path.mkdir(parents=True, exist_ok=True)

    # load & validate image+label pairs
    image_files = {
        f.stem: f
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    label_files = {
        f.stem: f
        for f in labels_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    }

    image_stems = set(image_files)
    label_stems = set(label_files)

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:
        raise ValueError(f"Missing labels for images: {sorted(missing_labels)}")

    if missing_images:
        raise ValueError(f"Missing images for labels: {sorted(missing_images)}")

    pairs: list[tuple[Path, Path]] = [
        (image_files[stem], label_files[stem])
        for stem in sorted(image_stems) # want to stop randomness here, set iteration order is arbitrary.
    ]

    if not pairs:
        raise ValueError("No valid image-label pairs found")

    if shuffle: # CONTINUE randomness here
        if seed is not None:
            random.seed(seed)
        random.shuffle(pairs)

    val_count = int(len(pairs) * val_frac)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    # create split directories
    for split in ["train", "val"]:
        for folder in ["images", "labels"]:
            (dest_path / split / folder).mkdir(parents=True, exist_ok=True)

    # copying files over to the new dataset in :DEST_PATH:
    for img, lbl in train_pairs:
        shutil.copy(img, dest_path / "train" / "images" / img.name)
        shutil.copy(lbl, dest_path / "train" / "labels" / lbl.name)

    for img, lbl in val_pairs:
        shutil.copy(img, dest_path / "val" / "images" / img.name)
        shutil.copy(lbl, dest_path / "val" / "labels" / lbl.name)

    print(f"  Train images: {len(train_pairs)}\n  Val images: {len(val_pairs)}")

    # now create the desired yaml file.
    # if we don't have "classes.txt" in the source, that's okay. we just tell the user.
    classes_path = source_path / "classes.txt"
    if classes_path.exists():
        with classes_path.open() as f:
            classes = [c.strip() for c in f.readlines()]

        names = {i: c for i, c in enumerate(classes) if c}

        # format data.yaml as a dictionary
        data = {
            "path": ".", # the yaml file ALWAYS lives alongside train/ and val/
            "train": "train/images",
            "val": "val/images",
            "names": names
        }

        with (dest_path / "data.yaml").open("w") as f:
            yaml.dump(data, f)

        print("Classes found, 'data.yaml' has been created.")
    else:
        print("No classes found, cannot create 'data.yaml'.")

# --- Code that takes all data and creates desired daughter datasets ---
def _generate_dataset_id(class_dirs: Iterable[Path]) -> str:
    """
    Generate a compact dataset ID of the form:
        idp_dataset_<yymmddhhmm>-n<N>-d<D>

    Where:
        - <yymmddhhmm> : timestamp
        - <N>           : total number of documents (images)
        - <D>           : number of document types (classes)
    """

    class_dirs = list(class_dirs)  # ensure multiple iteration
    num_doc_types = len(class_dirs)
    total_docs = sum(len(list((cls / "images").glob("*"))) for cls in class_dirs)

    timestamp = datetime.now().strftime("%Y-%m-%d@%H%M")

    dataset_id = f"dataset_{timestamp}_n{total_docs}d{num_doc_types}"
    return dataset_id

def idprocessing_train_dataset(
        source_path: str | Path,
        dest_path: str | Path = ".",
        effnet_ratios: tuple[float, float, float] = (0.8, 0.2, 0.0),
        yolo_val_frac: float = 0.2,
        create_merged_yolo: bool = False, # if we want another MERGED yolo dataset.
        seed: int | None = None,
        shuffle: bool = True
) -> None:
    """
    Use this function to create:
        1. EfficientNet doc-classifier dataset
        2. YOLO doc segmentation datasets

    Expected source structure:
        source_path/
        ├── doc_type_0/
        │   ├── images/
        │   ├── labels/
        │   └── classes.txt
        ├── doc_type_1/
        │   ├── images/
        │   ├── labels/
        │   └── classes.txt
        ...
        └── doc_type_k/
            ├── images/
            ├── labels/
            └── classes.txt
    :param source_path: Dataset Root 'path/to/data'
    :param dest_path: Dataset output
    :param effnet_ratios: EfficientNet split ratios (train, val, test)
    :param yolo_val_frac: YOLO validation split fraction
    :param create_merged_yolo: Allows us to create a large, merged YOLO dataset
    :param seed: Random seed
    :param shuffle: Choose whether to shuffle images (never really want to turn this off)
    :return: None
    """

    if seed is not None and isinstance(seed, int):
        random.seed(seed)

    # --------------------------------------------------
    # Resolve paths & validate folder structure
    # --------------------------------------------------

    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"{source_path} does not exist")

    doc_class_dirs = [d for d in source_path.iterdir() if d.is_dir()] # class folders
    if not doc_class_dirs:
        raise ValueError("No document type directories found")

    dest_path = Path(dest_path) / _generate_dataset_id(class_dirs=doc_class_dirs)
    print(f"Creating dataset from: {source_path}\nInto {dest_path}")

    summary: dict[str, Any] = {
        "dataset_name": dest_path.stem,
        "doc_types": [],
        "efficientnet": {
            "splits": {},
            "total_images": {}
        },
        "yolo_individual": {},
        "yolo_merged": None
    }

    class_sets: list[tuple[str, set[str]]] = []

    for cls in doc_class_dirs:
        cls_name = cls.name

        images_dir = cls / "images"
        labels_dir = cls / "labels"
        classes_file = cls / "classes.txt"

        if not images_dir.is_dir():
            raise FileNotFoundError(f"{cls_name} missing required folder: images/")

        if not labels_dir.is_dir():
            raise FileNotFoundError(f"{cls_name} missing required folder: labels/")

        if not classes_file.exists():
            raise FileNotFoundError(f"{cls_name} missing required file: classes.txt")

        # load segmentation class definitions
        with classes_file.open() as f:
            class_names = {
                line.strip()
                for line in f.readlines()
                if line.strip()
            }

        if not class_names:
            raise ValueError(f"{cls_name}/classes.txt is empty")

        class_sets.append((cls_name, class_names))

    canonical_classes = None

    for cls_name, class_set in class_sets:
        classes_file = source_path / cls_name / "classes.txt"
        with classes_file.open() as f:
            class_list = [line.strip() for line in f if line.strip()]

        if canonical_classes is None:
            canonical_classes = class_list
        else:
            if class_list != canonical_classes:
                raise ValueError(
                    f"YOLO class order mismatch in {cls_name}.\n"
                    f"Expected: {canonical_classes}\n"
                    f"Got: {class_list}"
                )

    # Warn if YOLO class definitions differ
    unique_sets = {frozenset(s) for _, s in class_sets}

    if len(unique_sets) > 1:
        print("(!) WARNING: Different segmentation class sets detected across doc types:")
        for name, s in class_sets:
            print(f"  - {name}: {sorted(s)}")
        print("Merged YOLO dataset (if created) will NOT remap class indices.")

    # ------------------------------------------------------------
    # Validate EfficientNet ratios
    # ------------------------------------------------------------

    train_r, val_r, test_r = effnet_ratios
    if round(train_r + val_r + test_r, 3) != 1.0:
        raise ValueError(f"EfficientNet ratios must sum to 1.0\nCurrent ratios (train, val, test) = {effnet_ratios}")

    split_ratios = {
        "train": train_r,
        "val": val_r,
        "test": test_r,
    }

    available_splits = [k for k, v in split_ratios.items() if v > 0]

    # --------------------------------------------------
    # 1. EfficientNet Dataset (split on doc type)
    # --------------------------------------------------

    print(f"Splitting documents into {available_splits}.")
    effnet_root = dest_path / "efficientnet_data"
    for split in available_splits:
        summary["efficientnet"]["splits"][split] = {}
        for cls in doc_class_dirs:
            (effnet_root / split / cls.name).mkdir(parents=True, exist_ok=True)

    for cls in doc_class_dirs:
        cls_name = cls.name
        summary["doc_types"].append(cls_name)

        images_dir = cls / "images"
        images = [
            p for p in images_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        if shuffle:
            random.shuffle(images)

        # handling splits
        N = len(images)
        print(f"Document Class: {cls_name}\n  Number of images: {N}")
        summary["efficientnet"]["total_images"][cls_name] = N

        n_train = int(N * train_r)
        n_val = int(N * val_r)

        idx = 0
        split_map = {}

        if train_r > 0:
            split_map["train"] = images[idx: idx + n_train]
            idx += n_train

        if val_r > 0:
            split_map["val"] = images[idx: idx + n_val]
            idx += n_val

        if test_r > 0:
            split_map["test"] = images[idx:]

        for split in available_splits:
            summary["efficientnet"]["splits"][split][cls_name] = len(split_map.get(split, []))
            for img in split_map.get(split, []):
                shutil.copy2(
                    img,
                    effnet_root / split / cls_name / img.name
                )

    # --------------------------------------------------
    # 2. Individual YOLO datasets using doc-type classes
    # --------------------------------------------------

    for cls in doc_class_dirs:
        cls_name = cls.name
        yolo_dest = dest_path / f"{cls_name}_yolo_data"
        print(f"Creating YOLO dataset for {cls_name}")

        _create_yolo_dataset(
            source_path=cls,
            dest_path=yolo_dest,
            val_frac=yolo_val_frac,
            shuffle=True,
            seed=seed
        )

        num_images = len([
            p for p in yolo_dest.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        num_labels = len([
            p for p in yolo_dest.rglob("*.txt") if "labels" in str(p.parent)
        ])

        summary["yolo_individual"][cls_name] = {
            "images": num_images,
            "labels": num_labels,
            "classes": canonical_classes
        }

    # --------------------------------------------------
    # 3. OPTIONAL merged YOLO dataset with
    # all images spread evenly across doc types
    # --------------------------------------------------

    if create_merged_yolo:
        print(f"Creating merged YOLO dataset for all document types")

        # create a TEMPORARY load of data inside dest/_merged_yolo_source
        merged_source = dest_path / "_merged_yolo_source"
        (merged_source / "images").mkdir(parents=True, exist_ok=True)
        (merged_source / "labels").mkdir(parents=True, exist_ok=True)

        for cls in doc_class_dirs:
            images_dir = cls / "images"
            labels_dir = cls / "labels"

            for img in images_dir.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl = labels_dir / f"{img.stem}.txt"
                if not lbl.exists():
                    raise ValueError(
                        f"Missing label for image {img.name} in {cls.name}"
                    )

                shutil.copy2(img, merged_source / "images" / img.name)
                shutil.copy2(lbl, merged_source / "labels" / lbl.name)

        with (merged_source / "classes.txt").open("w") as f:
            for name in canonical_classes:
                f.write(name + "\n")

        merged_dest = dest_path / "merged_yolo_dataset"
        _create_yolo_dataset(
            source_path=merged_source,
            dest_path=merged_dest,
            val_frac=yolo_val_frac,
            shuffle=True,
            seed=seed
        )

        num_images = len([
            p for p in merged_dest.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        num_labels = len([
            p for p in merged_dest.rglob("*.txt") if "labels" in str(p.parent)
        ])

        summary["yolo_merged"] = {
            "images": num_images,
            "labels": num_labels,
            "classes": canonical_classes
        }

    # Dump summary into JSON
    summary_file = dest_path / "info.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary written to {summary_file}")

if __name__ == "__main__":
    print(idprocessing_train_dataset("../all_images/new_yolo_data", create_merged_yolo=True))