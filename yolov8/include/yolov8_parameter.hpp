#pragma once

#include <array>
#include <string>

namespace yolov8 {

namespace parameter {




struct Parameter {
  struct image_info {
    int h{224};
    int w{224};
    int c{3};
  };
  image_info img_info;
  // TODO
  std::array<int, 3> output_dims{1, 8400, 84};
};



static std::array<std::string, 80> kCocoLabel = {
  "person",     "bicycle",    "car",        "motorcycle", "airplane", "bus",       "train",        "truck",        "boat",         "traffic light",  "fire hydrant", "stop sign", "parking meter",
  "bench",      "bird",       "cat",        "dog",        "horse",    "sheep",     "cow",          "elephant",     "bear",         "zebra",          "giraffe",      "backpack",  "umbrella",
  "handbag",    "tie",        "suitcase",   "frisbee",    "skis",     "snowboard", "sports ball",  "kite",         "baseball bat", "baseball glove", "skateboard",   "surfboard", "tennis racket",
  "bottle",     "wine glass", "cup",        "fork",       "knife",    "spoon",     "bowl",         "banana",       "apple",        "sandwich",       "orange",       "broccoli",  "carrot",
  "hot dog",    "pizza",      "donut",      "cake",       "chair",    "couch",     "potted plant", "bed",          "dining table", "toilet",         "tv",           "laptop",    "mouse",
  "remote",     "keyboard",   "cell phone", "microwave",  "oven",     "toaster",   "sink",         "refrigerator", "book",         "clock",          "vase",         "scissors",  "teddy bear",
  "hair drier", "toothbrush"};

}   // namespace parameter

}   // namespace yolov8