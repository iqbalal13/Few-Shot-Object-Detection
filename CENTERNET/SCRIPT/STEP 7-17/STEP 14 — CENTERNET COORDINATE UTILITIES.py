# ============================================================
# STEP 14 — CENTERNET COORDINATE UTILITIES
# ============================================================

print("=" * 70)
print("STEP 14 — CENTERNET COORDINATE UTILITIES")
print("=" * 70)

OUTPUT_STRIDE = CONFIG["output_stride"]

OUTPUT_HEIGHT = (
    CONFIG["input_size"]
    // OUTPUT_STRIDE
)

OUTPUT_WIDTH = (
    CONFIG["input_size"]
    // OUTPUT_STRIDE
)


def xyxy_to_center_wh(box):
    """
    Input:
        [x1, y1, x2, y2]

    Output:
        center_x, center_y, width, height
    """

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    return (
        center_x,
        center_y,
        width,
        height
    )


def image_to_output_coordinates(
    center_x,
    center_y,
    width,
    height,
    stride=OUTPUT_STRIDE
):

    return (
        center_x / stride,
        center_y / stride,
        width / stride,
        height / stride
    )


print("Input resolution  :",
      CONFIG["input_size"],
      "x",
      CONFIG["input_size"])

print("Output stride     :",
      OUTPUT_STRIDE)

print("Output resolution :",
      OUTPUT_HEIGHT,
      "x",
      OUTPUT_WIDTH)

assert OUTPUT_HEIGHT == 160
assert OUTPUT_WIDTH == 160

test_box = torch.tensor(
    [100., 120., 300., 400.]
)

cx, cy, w, h = xyxy_to_center_wh(
    test_box
)

cx_o, cy_o, w_o, h_o = \
    image_to_output_coordinates(
        cx, cy, w, h
    )

print("\nTest bbox:")
print("Input center :", float(cx), float(cy))
print("Input WH     :", float(w), float(h))

print("Output center:",
      float(cx_o),
      float(cy_o))

print("Output WH    :",
      float(w_o),
      float(h_o))

print("\nSTEP 14 PASSED")
