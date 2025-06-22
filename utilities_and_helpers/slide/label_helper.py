import json

#produces dictionaries from annotation files that map category names to category ids
#but the "real" code decides the actual value mappings by setting attributes of slide contrainers through load_slides()

# Load label dictionary from annotation file
def load_label_dict(annotation_file):
    with open(annotation_file) as f:
        data = json.load(f)
    label_dict = {cat["name"]: cat["id"] for cat in data["categories"]}
    label_dict["Background"] = 0
    label_dict["Unassigned"] = -1
    return label_dict


# Reverse label dictionary for visualization
def reverse_label_dict(label_dict):
    reversed_dict = {}
    for k, v in label_dict.items():
        if v not in reversed_dict:
            reversed_dict[v] = [k]
        else:
            reversed_dict[v].append(k)
    return reversed_dict
