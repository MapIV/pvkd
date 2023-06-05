labels = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}

color_map = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [255, 255, 255],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0]
}

# color_map = {
#     0: [0, 0, 0],  # outlier
#     1: [0, 0, 142],  # car
#     2: [119, 11, 32],  # bicycle
#     3: [220, 20, 60],  # person
#     4: [0, 0, 70],  # truck
#     5: [128, 64, 128],  # road
#     6: [244, 35, 232],  # other-ground
#     7: [128, 64, 64],  # other_vehicle
#     8: [152, 251, 152],  # terrain
#     9: [70, 70, 70],  # building
#     10: [190, 153, 153],  # fence
#     11: [107, 142, 35],  # vegetation
#     12: [255, 255, 255],  # moving
#     13: [220, 220, 0],  # traffic-sign
# }

color_map = {
    0: [70, 70, 70],  # outlier
    1: [128, 64, 128],  # road
    2: [244, 35, 232],  # bicycle
    3: [220, 20, 60],  # person
    4: [152, 251, 152],  # truck
}