import os
import json

root_path = '/public/home/zhaoyq'


THREED_FRONT_BEDROOM_FURNITURE = {
    "desk":                                    "desk",
    "nightstand":                              "nightstand",
    "king-size bed":                           "double_bed",
    "single bed":                              "single_bed",
    "kids bed":                                "kids_bed",
    "bookcase/jewelry armoire":                "bookshelf",
    "tv stand":                                "tv_stand",
    "wardrobe":                                "wardrobe",
    "lounge chair/cafe chair/office chair":    "chair",
    "dining chair":                            "chair",
    "classic chinese chair":                   "chair",
    "armchair":                                "armchair",
    "dressing table":                          "dressing_table",
    "dressing chair":                          "dressing_chair",
    "corner/side table":                       "table",
    "dining table":                            "table",
    "round end table":                         "table",
    "drawer chest/corner cabinet":             "cabinet",
    "sideboard/side cabinet/console table":    "cabinet",
    "children cabinet":                        "children_cabinet",
    "shelf":                                   "shelf",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",#????
    "coffee table":                            "coffee_table",
    "loveseat sofa":                           "sofa",
    "three-seat/multi-seat sofa":              "sofa",
    "l-shaped sofa":                           "sofa",
    "lazy sofa":                               "sofa",
    "chaise longue sofa":                      "sofa",
}


THREED_FRONT_LIBRARY_FURNITURE = {
    "bookcase/jewelry armoire":                "bookshelf",
    "desk":                                    "desk",
    "lounge chair/cafe chair/office chair":    "lounge_chair",
    "dining chair":                            "dining_chair",
    "dining table":                            "dining_table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chinese_chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard/side cabinet/console table":    "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "round_end_table",
    "loveseat sofa":                           "loveseat_sofa",
    "drawer chest/corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat/multi-seat sofa":              "multi_seat_sofa",
    "wine cabinet":                            "wine_cabinet", ##!!!
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "lazy_sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "chaise_longue_sofa",
    "l-shaped sofa":                           "l_shaped_sofa",
    "dressing table":                          "dressing_table",
    "dressing chair":                          "dressing_chair",
}

THREED_FRONT_LIVINGROOM_FURNITURE = {
    "bookcase/jewelry armoire":                "bookshelf",
    "desk":                                    "desk",
    "lounge chair/cafe chair/office chair":    "lounge_chair",
    "dining chair":                            "dining_chair",
    "dining table":                            "dining_table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chinese_chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard/side cabinet/console table":    "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "round_end_table",
    "loveseat sofa":                           "loveseat_sofa",
    "drawer chest/corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat/multi-seat sofa":              "multi_seat_sofa",
    "wine cabinet":                            "wine_cabinet",
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "lazy_sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "chaise_longue_sofa",
    "l-shaped sofa":                           "l_shaped_sofa",
    "tv stand":                                "tv_stand"
}

import random

def get_used():
    with open(os.path.join(root_path, '3D-FUTURE-model_sdf', 'used.csv')) as f:
        rows = [row.strip() for row in f]
    f.close()
    st = set(rows)
    with open(os.path.join(root_path, '3D-FUTURE-model_sdf', 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    f.close()

    for_eval = []
    for item in model_info:
        if item['category'] is None or item['model_id'].lower() in st:continue

        if item['category'].lower() in THREED_FRONT_BEDROOM_FURNITURE.keys():
            for_eval.append(item['model_id'].lower())


    eval_set = random.SystemRandom().sample(for_eval, 100)
    print(len(for_eval))

    with open(os.path.join(root_path,'3D-FUTURE-model_sdf', 'eval.csv'), 'w') as f:
        for eval in eval_set:
            f.writelines(eval+'\n')


get_used()

