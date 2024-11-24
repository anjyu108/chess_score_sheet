import argparse
import cv2
import os


# These parameter is for NCS score sheet
NOTATION_AREA_UPPER = 260
NOTATION_AREA_LOWER = 1800
NOTATION_AREA_LEFT = 150
NOTATION_AREA_RIGHT = 1280

MOVE_ID_WIDTH = 50


def split_sheet_image_into_move(sheet_image_path, out_dir_path, move_num):
    # load data
    sheet_image = cv2.imread(sheet_image_path, cv2.IMREAD_GRAYSCALE)
    if sheet_image is None:
        print("Error: sheet_image is None")
        return

    move_rect_list = []  # (x1, y1, x2, y2)
    img_height, img_width = sheet_image.shape

    sheet_image = sheet_image[NOTATION_AREA_UPPER:NOTATION_AREA_LOWER,
                              NOTATION_AREA_LEFT:NOTATION_AREA_RIGHT]
    img_height, img_width = sheet_image.shape

    # split sheet into move
    move_height, move_width = (img_height // 30), (((img_width // 2) - MOVE_ID_WIDTH) // 2)
    move_height_margin = int(move_height * 0.1)
    # TODO: refactor this block
    # NOTE: why y margin bigger? <- Player used to write move near lower line
    move_num_on_sheet = 60
    for move_idx in range(move_num_on_sheet):
        if move_idx < 30:
            # white move
            x1, y1 = 0, move_height * move_idx
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            x1, x2 = x1 + MOVE_ID_WIDTH, x2 + MOVE_ID_WIDTH
            move_rect_list.append((x1, y1, x2, y2))

            # black move
            x1, y1 = move_width, move_height * move_idx
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            x1, x2 = x1 + MOVE_ID_WIDTH, x2 + MOVE_ID_WIDTH
            move_rect_list.append((x1, y1, x2, y2))
        else:
            # white move
            x1, y1 = (img_width // 2), move_height * (move_idx - 30)
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            x1, x2 = x1 + MOVE_ID_WIDTH, x2 + MOVE_ID_WIDTH
            move_rect_list.append((x1, y1, x2, y2))

            # black move
            x1, y1 =  (img_width // 2) + move_width, move_height * (move_idx - 30)
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            x1, x2 = x1 + MOVE_ID_WIDTH, x2 + MOVE_ID_WIDTH
            move_rect_list.append((x1, y1, x2, y2))

    # Split and output images for each move
    os.makedirs(out_dir_path, exist_ok=True)
    move_image_path_list = []
    for move_idx in range(move_num_on_sheet * 2):
        if move_idx >= move_num:
            break

        x1, y1, x2, y2 = move_rect_list[move_idx]
        img_cropped = sheet_image[y1:y2, x1:x2]

        # for make it easy to sort, add 0, 1 prefix
        color = "0white" if move_idx % 2 == 0 else "1black"
        move_name_idx = (move_idx // 2) + 1
        move_name = "{:0=4}".format(move_name_idx) + "_" + color
        move_image_path = os.path.join(out_dir_path, move_name + '.png')
        move_image_path_list.append(move_image_path)
        cv2.imwrite(move_image_path, img_cropped)

    return move_image_path_list


def main(args):
    split_sheet_image_into_move(args.input, args.output, args.move_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="data/kanagawa_champ_2023/images/1.jpeg")
    parser.add_argument('--output', default="data/kanagawa_champ_2023/images_move/1/")
    parser.add_argument('--move_num', type=int, default=60)
    args = parser.parse_args()

    main(args)
