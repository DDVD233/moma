import os
import ujson as json
from momaapi import MOMA, AnnVisualizer
from tqdm import tqdm
from detectron2.structures import BoxMode


def get_categories(moma):
    cids = moma.get_cids("actor", 0, "either")
    cid_to_cname = {cid: moma.get_cnames(cids_actor=[cid])[0] for cid in cids}
    cids_objects = moma.get_cids("object", 0, "either")
    # +26 to avoid collision with actor cids
    cid_to_cname.update({cid + 26: moma.get_cnames(cids_object=[cid])[0] for cid in cids_objects})
    categories = []

    for cid, name in cid_to_cname.items():
        categories.append({"id": cid+1, "name": name, "supercategory": "none"})
    return categories


def create_dataset(moma, ids_hoi, categories):
    """
    Prepares the dataset in COCO format.
    """
    coco = {"images": [], "annotations": [], "categories": categories}
    cname_to_cid = {category["name"]: category["id"] for category in categories}
    dir_moma = moma.dir_moma

    for id_hoi in ids_hoi:
        ann_hoi = moma.get_anns_hoi([id_hoi])[0]
        image_path = moma.get_paths(ids_hoi=[id_hoi])[0]
        relative_path = os.path.relpath(image_path, dir_moma)
        id_act = moma.get_ids_act(ids_hoi=[id_hoi])[0]
        metadatum = moma.get_metadata(ids_act=[id_act])[0]

        coco["images"].append({
            "file_name": relative_path,
            "id": len(coco["images"]),
            "width": metadatum.width,
            "height": metadatum.height
        })

        entities = ann_hoi.actors + ann_hoi.objects

        for entity in entities:
            if entity.cname in cname_to_cid:
                annotation = {
                    "bbox": [
                        entity.bbox.x,
                        entity.bbox.y,
                        entity.bbox.width,
                        entity.bbox.height,
                    ],
                    "category_id": cname_to_cid[entity.cname],
                    "image_id": len(coco["images"]) - 1,
                    "id": len(coco["annotations"]),
                    "iscrowd": 0,
                    "area": entity.bbox.width * entity.bbox.height,
                }
                coco["annotations"].append(annotation)

    return coco


if __name__ == "__main__":
    dir_moma = "/media/dvd/Others/moma"
    moma = MOMA(dir_moma)

    train_hoi_ids = moma.get_ids_hoi(split="train")
    val_hoi_ids = moma.get_ids_hoi(split="val")
    categories = get_categories(moma)

    train_coco = create_dataset(moma, train_hoi_ids, categories)
    val_coco = create_dataset(moma, val_hoi_ids, categories)

    with open(os.path.join(dir_moma, "coco_train.json"), "w") as f:
        json.dump(train_coco, f)

    with open(os.path.join(dir_moma, "coco_val.json"), "w") as f:
        json.dump(val_coco, f)