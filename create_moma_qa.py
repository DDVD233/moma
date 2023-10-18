import random
import string
import math

from momaapi import MOMA, AnnVisualizer
from tqdm import tqdm
import ujson as json
import os
from collections import defaultdict
from momaapi.visualizers import AnnVisualizer

dir_moma = "/media/dvd/Others/moma"
# dir_moma = "/Volumes/Others/moma"
moma = MOMA(dir_moma)


def compute_distance_or_iou(bbox1, bbox2):
    '''Compute the Intersection over Union (IoU) of two bounding boxes.
    If IoU is 0, compute the negative Euclidean distance between their centers.'''

    x1, y1, x2, y2 = bbox1
    x1_int, y1_int, x2_int, y2_int = bbox2

    # Determine the coordinates of the intersection rectangle
    x_left = max(x1, x1_int)
    y_top = max(y1, y1_int)
    x_right = min(x2, x2_int)
    y_bottom = min(y2, y2_int)

    # Intersection area
    inter_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    if inter_area == 0:
        # Compute the centers of the bounding boxes
        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        center2 = ((x1_int + x2_int) / 2, (y1_int + y2_int) / 2)

        # Compute the Euclidean distance between the centers and return its negative value
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return -distance

    # Area of both rectangles
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_int - x1_int) * (y2_int - y1_int)

    # Union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou


def generate_random_identity(size=5):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(size))


def match_objects(list1, list2, iou_threshold=-float('inf')):
    # Sort the objects by xmin
    list1 = dict(sorted(list1.items(), key=lambda item: item[1]['bbox'][0]))
    list2 = dict(sorted(list2.items(), key=lambda item: item[1]['bbox'][0]))
    mapping = {}
    unmatched = set(list2.keys())
    for key1, obj1 in list1.items():
        max_iou = -float('inf')
        best_match_key = None
        for key2, obj2 in list2.items():
            if obj1['name'] == obj2['name'] and key2 in unmatched:
                iou = compute_distance_or_iou(obj1['bbox'], obj2['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_match_key = key2
        if best_match_key and max_iou > iou_threshold:
            mapping[key1] = best_match_key
            unmatched.remove(best_match_key)
        else:
            mapping[key1] = generate_random_identity()
    return mapping


def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def map_ids(act):
    sact_ids = act.ids_sact
    sacts = moma.get_anns_sact(ids_sact=sact_ids)

    # Real ID: Identity consistent across sacts
    # Original ID: Identity in the original annotation, may not be consistent across sacts
    real_to_original = defaultdict(dict)  # {real_id: {sact_id: original_id}}
    original_to_real = defaultdict(dict)  # {sact_id: {original_id: real_id}}
    previous_identity_boxes = {}

    for sact_index, sact in enumerate(sacts):
        last_hoi = sact.ids_hoi[-1]
        last_ann_hoi = moma.get_anns_hoi(ids_hoi=[last_hoi])[0]
        first_hoi = sact.ids_hoi[0]
        first_ann_hoi = moma.get_anns_hoi(ids_hoi=[first_hoi])[0]
        boxes = [get_boxes(ann_hoi) for ann_hoi in moma.get_anns_hoi(ids_hoi=sact.ids_hoi)]
        all_boxes = {}

        for box_list in boxes:
            for identity, data in box_list.items():
                if identity not in all_boxes:
                    all_boxes[identity] = data

        if len(previous_identity_boxes) == 0:
            # This is the first sact. Map the identities from the last box to the real id.
            for identity, value in all_boxes.items():
                real_to_original[identity][sact.id] = identity
                original_to_real[sact.id][identity] = identity
        elif len(previous_identity_boxes) > 0:  # For clarity
            # Map the identities from the last box of the previous sact to the first box of the current sact
            # identity_mapping: real_id -> original_id
            identity_mapping = match_objects(all_boxes, previous_identity_boxes)
            for original_id, real_id in identity_mapping.items():
                real_to_original[real_id][sact.id] = original_id
                original_to_real[sact.id][original_id] = real_id
        last_boxes = {}
        for box_list in boxes[::-1]:  # Now find the last occurence of each identity
            for identity, data in box_list.items():
                if identity not in last_boxes:
                    last_boxes[identity] = data
        # for previous identity boxes key, make it real id
        keys = list(last_boxes.keys())
        previous_identity_boxes = {}
        for key in keys:
            if key not in original_to_real[sact.id]:
                # This identity is newly introduced in this sact
                raise Exception("This should not happen")
            previous_identity_boxes[original_to_real[sact.id][key]] = last_boxes.pop(key)

    return real_to_original, original_to_real


def get_boxes(ann_hoi):
    current_identity_boxes = {}
    for actor in ann_hoi.actors:
        current_identity_boxes[actor.id] = {
            "bbox": [actor.bbox.x, actor.bbox.y, actor.bbox.x2, actor.bbox.y2],
            "name": actor.cname
        }
    for object_ in ann_hoi.objects:
        current_identity_boxes[object_.id] = {
            "bbox": [object_.bbox.x, object_.bbox.y, object_.bbox.x2, object_.bbox.y2],
            "name": object_.cname
        }
    return current_identity_boxes


def create_identity_mapping(split):
    train_act_ids = moma.get_ids_act(split=split)
    train_act_anns = moma.get_anns_act(train_act_ids)

    identity_mappings = {}

    for act in tqdm(train_act_anns):
        act_id = act.id
        real_to_original, original_to_real = map_ids(act)
        identity_mappings[act_id] = (real_to_original, original_to_real)

    return identity_mappings


def create_moma_qa(split, identity_mappings):
    train_act_ids = moma.get_ids_act(split=split)
    train_act_anns = moma.get_anns_act(train_act_ids)

    annotations = []
    rel_count = 0
    sact_count = 0
    basic_count = 0
    question_count = 0
    bbox_question_count = 0
    dir_vis = os.path.join(dir_moma, 'videos')
    visualizer = AnnVisualizer(moma, dir_vis=dir_vis)

    for act in tqdm(train_act_anns):
        sact_ids = act.ids_sact
        act_id = act.id
        sacts = moma.get_anns_sact(ids_sact=sact_ids)
        video_metadata = moma.get_metadata(ids_act=[act_id])[0]
        video_id = video_metadata.fname.replace('.mp4', '')
        captions = [sact.cname for sact in sacts]

        # Real ID: Identity consistent across sacts
        # Original ID: Identity in the original annotation, may not be consistent across sacts
        # real_to_original: {real_id: {sact_id: original_id}}
        # original_to_real: {sact_id: {original_id: real_id}}
        real_to_original, original_to_real = identity_mappings[act_id]

        for sact_index, sact in enumerate(sacts):
            sact_count += 1
            ids_hoi = sact.ids_hoi
            anns_hoi = moma.get_anns_hoi(ids_hoi=ids_hoi)
            question_to_index = {}

            caption = sact.cname
            # Find out how many times has the exact same caption appeared
            caption_count = captions[:sact_index + 1].count(caption)
            # add a "for the nth time" to the caption
            caption_ordinal = make_ordinal(caption_count)

            for ann_hoi in anns_hoi:
                things_to_name = map_things_to_name(ann_hoi)

                node_descriptions = []
                for actor in ann_hoi.actors:
                    node_descriptions.append(make_node_description(ann_hoi, actor))

                num_actors = len(ann_hoi.actors)
                actor_question = f"When {caption}, how many people are in the scene?"
                if actor_question in question_to_index:
                    if str(num_actors) != annotations[question_to_index[actor_question]]["answer"]:
                        original_answer = int(annotations[question_to_index[actor_question]]["answer"])
                        annotations[question_to_index[actor_question]]["answer"] = str(max(original_answer, num_actors))
                else:
                    basic_count += 1
                    annotations.append({
                        "question": actor_question,
                        "question_id": question_count,
                        "answer": str(num_actors),
                        "filename": f"raw/{video_metadata.fname}",
                        "video": f"raw/{video_metadata.fname}",
                        "video_id": f"raw/{video_id}",
                        "height": video_metadata.height,
                        "width": video_metadata.width,
                        "num_frames": video_metadata.num_frames,
                        "sact_start": sact.start,
                        "sact_end": sact.end,
                        "sgg_question": actor_question,
                        "duration": int(video_metadata.duration)
                    })
                    question_count += 1
                    question_to_index[actor_question] = len(annotations) - 1

                for rel in ann_hoi.rels:
                    question = rel.cname
                    # If answer id is number->object, else->actor
                    assert isinstance(rel.id_trg, str)

                    video_filename = f"raw/{video_metadata.fname}"
                    this_video_id = video_filename.replace('.mp4', '')
                    description_for_sact = make_sact_descriptions(ann_hoi, things_to_name)

                    # Draw the bbox for the source node if it is indistinguishable
                    source_node = things_to_name[rel.id_src]
                    this_node_description = make_node_description(ann_hoi, source_node)
                    if node_descriptions.count(this_node_description) > 1:
                        # This node is indistinguishable for the question
                        this_real_id = original_to_real[sact.id][rel.id_src]
                        this_original_ids = real_to_original[this_real_id]
                        video_filename = visualizer.draw_bbox(id_act=act.id, id_sact=sact.id, real_id=this_real_id,
                                                              original_ids=this_original_ids,
                                                              duration=video_metadata.duration)
                        this_video_id = video_filename.replace('.mp4', '')
                        anonymized_name = "person" if source_node.kind == "actor" else "object"
                        this_node_description = f"highlighted {anonymized_name}"
                        description_for_sact = make_sact_descriptions(ann_hoi, things_to_name,
                                                                      highlight_target=rel.id_src)
                        bbox_question_count += 1

                    # QA part
                    interrog_word = "What" if rel.id_trg.isnumeric() else "Who"
                    question = question.replace("[src]", f"{interrog_word} is the {this_node_description}")
                    question = question.replace(" [trg]", f"")
                    question = f"When {caption} for the {caption_ordinal} time, {question}?"
                    answer = things_to_name[rel.id_trg].cname

                    question = question.replace("unclassified ", "")
                    answer = answer.replace("unclassified ", "")

                    if answer == "unsure":
                        continue

                    sgg_question = '. '.join(description_for_sact) + f". {question}"
                    sgg_question = sgg_question.replace("unclassified ", "")
                    annotation = {
                        "question": question,
                        "question_id": question_count,
                        "answer": answer,
                        "filename": video_filename,
                        "video": video_filename,
                        "video_id": this_video_id,
                        "height": video_metadata.height,
                        "width": video_metadata.width,
                        "num_frames": video_metadata.num_frames,
                        "sact_start": sact.start,
                        "sact_end": sact.end,
                        "sgg_question": sgg_question,
                        "duration": int(video_metadata.duration)
                    }
                    annotations.append(annotation)
                    question_to_index[question] = len(annotations) - 1
                    rel_count += 1
                    question_count += 1

    print(f"Total number of annotations: {len(annotations)}")
    print(f"Total number of sacts: {sact_count}")
    print(f"Total number of rels: {rel_count}")
    print(f"Total number of bbox questions: {bbox_question_count}")
    print(f"Total number of basic questions: {basic_count}")

    return annotations


def make_node_description(ann_hoi, source_node):
    attributes = ann_hoi.ias
    if isinstance(ann_hoi.atts, list):
        attributes.extend(ann_hoi.atts)
    source_node_ia = list(filter(lambda att: att.id_src == source_node.id, attributes))
    if len(source_node_ia) == 0:
        return source_node.cname
    ia_attribute = source_node_ia[0].cname.replace('[src] ', '')
    return f"{ia_attribute} {source_node.cname}"


def map_things_to_name(ann_hoi):
    things_to_name = {}
    for actor in ann_hoi.actors:
        things_to_name[actor.id] = actor
    for object in ann_hoi.objects:
        things_to_name[object.id] = object
    return things_to_name


def make_sact_descriptions(ann_hoi, things_to_name, highlight_target=None):
    description_for_sact = []
    for rel in ann_hoi.rels:
        # If answer id is number->object, else->actor
        assert isinstance(rel.id_trg, str)

        source_name = things_to_name[rel.id_src].cname
        if highlight_target is not None and rel.id_src == highlight_target:
            source_name = f"highlighted {source_name}"

        # sg part
        sg_caption = rel.cname
        sg_caption = sg_caption.replace('[src]', f'{source_name} is')
        sg_caption = sg_caption.replace('[trg]',
                                        f'{things_to_name[rel.id_trg].cname}')  # player is beneath basketball hoop
        if sg_caption not in description_for_sact:
            description_for_sact.append(sg_caption)
    return description_for_sact


if __name__ == "__main__":
    save_path = '/home/data/datasets/moma_qa/'

    # identity_mapping_train = create_identity_mapping(split="train")
    # identity_mapping_val = create_identity_mapping(split="val")
    # identity_mapping_test = create_identity_mapping(split="test")
    # # combine all identity mappings
    # identity_mapping_all = identity_mapping_train.copy()
    # identity_mapping_all.update(identity_mapping_val)
    # identity_mapping_all.update(identity_mapping_test)
    #
    # with open(os.path.join(save_path, 'identity_mapping.json'), 'w') as f:
    #     json.dump(identity_mapping_all, f)

    # Load identity mapping
    with open(os.path.join(save_path, 'identity_mapping.json'), 'r') as f:
        identity_mapping_all = json.load(f)

    annotations_train = create_moma_qa(split="train", identity_mappings=identity_mapping_all)
    with open(os.path.join(save_path, 'train.json'), 'w') as f:
        json.dump(annotations_train, f)
    annotations_val = create_moma_qa(split="val", identity_mappings=identity_mapping_all)
    annotations_test = create_moma_qa(split="test", identity_mappings=identity_mapping_all)
    annotations_all = annotations_train + annotations_val + annotations_test



    with open(os.path.join(save_path, 'val.json'), 'w') as f:
        json.dump(annotations_val, f)
    with open(os.path.join(save_path, 'test.json'), 'w') as f:
        json.dump(annotations_val, f)

    with open(os.path.join(save_path, 'all.json'), "w") as f:
        json.dump(annotations_all, f)
