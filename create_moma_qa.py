from momaapi import MOMA, AnnVisualizer
from tqdm import tqdm
import ujson as json
import os
from collections import defaultdict
from momaapi.visualizers import AnnVisualizer

dir_moma = "/media/dvd/Others/moma"
moma = MOMA(dir_moma)


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


def create_moma_qa(split):
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
                        "video_id": video_id,
                        "height": video_metadata.height,
                        "width": video_metadata.width,
                        "num_frames": video_metadata.num_frames,
                        "sact_start": sact.start,
                        "sact_end": sact.end,
                        "sgg_question": actor_question
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
                        video_filename = visualizer.draw_bbox(id_act=act.id, id_entity=rel.id_src, id_sact=sact.id)
                        this_video_id = video_filename.replace('.mp4', '')
                        anonymized_name = "person" if source_node.kind == "actor" else "object"
                        this_node_description = f"highlighted {anonymized_name}"
                        description_for_sact = make_sact_descriptions(ann_hoi, things_to_name,
                                                                      highlight_target=rel.id_src)
                        bbox_question_count += 1

                    # QA part
                    interrog_word = "what" if rel.id_trg.isnumeric() else "who"
                    question = question.replace("[src]", f"{interrog_word} is the {this_node_description}")
                    question = question.replace(" [trg]", f"")
                    question = f"When {caption} for the {caption_ordinal} time, {question}?"
                    answer = things_to_name[rel.id_trg].cname

                    if answer == "unsure":
                        continue

                    sgg_question = '. '.join(description_for_sact) + f". {question}"
                    annotation = {
                        "question": question,
                        "question_id": question_count,
                        "answer": answer,
                        "filename": video_filename,
                        "video_id": this_video_id,
                        "height": video_metadata.height,
                        "width": video_metadata.width,
                        "num_frames": video_metadata.num_frames,
                        "sact_start": sact.start,
                        "sact_end": sact.end,
                        "sgg_question": sgg_question
                    }
                    annotations.append(annotation)
                    question_to_index[question] = len(annotations) - 1
                    rel_count += 1

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
        sg_caption = sg_caption.replace('[trg]', f'{things_to_name[rel.id_trg].cname}')  # player is beneath basketball hoop
        if sg_caption not in description_for_sact:
            description_for_sact.append(sg_caption)
    return description_for_sact


if __name__ == "__main__":
    save_path = '/home/data/datasets/moma_qa/'
    annotations_train = create_moma_qa(split="train")
    with open(os.path.join(save_path, 'train.json'), 'w') as f:
        json.dump(annotations_train, f)
    annotations_val = create_moma_qa(split="val")
    annotations_test = create_moma_qa(split="test")
    annotations_all = annotations_train + annotations_val + annotations_test

    with open(os.path.join(save_path, 'val.json'), 'w') as f:
        json.dump(annotations_val, f)
    with open(os.path.join(save_path, 'test.json'), 'w') as f:
        json.dump(annotations_val, f)

    with open(os.path.join(save_path, 'all.json'), "w") as f:
        json.dump(annotations_all, f)