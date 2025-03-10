import argparse
from pprint import pprint
import cv2
import pdb
import os


def get_data_sheet(args):
    """
    Load all the data
    Args:
        args: list of image path (1 path = 1 sheet), list of labels (1 label = 1 sheet)
              (label one or multple rows, and 1 row = 1 move:w)
    """
    training_tags = open(f"{args.input}/training_tags.txt", "r").readlines()

    ret = {}
    for line in training_tags:
        line = line.strip()
        game_id, sheet_id, move_id, color = line.split('_')
        color = color.split(" ")[0].strip(".png")
        move = line.split(' ')[-1]
        if game_id not in ret.keys():
            ret[game_id] = {}
        if sheet_id not in ret[game_id].keys():
            ret[game_id][sheet_id] = {}
        move_name = move_id + "_" + color
        ret[game_id][sheet_id][move_name] = move

    return ret


def get_data_move(data_sheet, args):
    """
    Load all the data
    Args:
        args: list of image path (1 path = 1 sheet), list of labels (1 label = 1 sheet)
              (labiel one or multple rows, and 1 row = 1 move:w)
    """
    # pdb.set_trace()
    image_path_list = []  # 1 image = 1 move(=part of score sheet image)
    move_list = []

    for game_id in data_sheet.keys():
        for sheet_id in data_sheet[game_id].keys():
            sheet_image_filename = game_id + "_" + sheet_id + ".png"
            sheet_image_path = args.input + "/" + sheet_image_filename
            move_image_path_list = split_sheet_image_into_move(sheet_image_path)
            for move_idx, move_image_path in enumerate(move_image_path_list):
                color = "white" if move_idx % 2 == 0 else "black"
                move_name = str((move_idx // 2) + 1) + "_" + color
                if data_sheet[game_id][sheet_id].get(move_name, None) is None:
                    continue
                move_list.append(data_sheet[game_id][sheet_id][move_name])
                image_path_list.append(move_image_path)

    assert len(image_path_list) == len(move_list)
    return image_path_list, move_list


def split_sheet_image_into_move(sheet_image_path):
    # load data
    sheet_image = cv2.imread(sheet_image_path, cv2.IMREAD_GRAYSCALE)
    if sheet_image is None:
        return []

    # split sheet into move
    move_rect_list = []  # (x1, y1, x2, y2)
    img_height, img_width = sheet_image.shape
    move_height, move_width = (img_height // 30), (img_width // 4)
    move_height_margin = int(move_height * 0.1)
    # TODO: refactor this block
    # NOTE: why y margin bigger? <- Player used to write move near lower line
    move_num_on_sheet = 60 * 2
    for move_idx in range(move_num_on_sheet):
        if move_idx < 30:
            # white move
            x1, y1 = 0, move_height * move_idx
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            move_rect_list.append((x1, y1, x2, y2))

            # black move
            x1, y1 = move_width, move_height * move_idx
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            move_rect_list.append((x1, y1, x2, y2))
        else:
            # white move
            x1, y1 = move_width * 2, move_height * (move_idx - 30)
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            move_rect_list.append((x1, y1, x2, y2))

            # black move
            x1, y1 = move_width * 3, move_height * (move_idx - 30)
            x2, y2 = x1 + move_width, y1 + move_height
            y1 = max(0, y1 - move_height_margin)
            y2 = min(img_height, y2 + move_height_margin * 2)
            move_rect_list.append((x1, y1, x2, y2))

    # Split and output images for each move

    move_image_dir = sheet_image_path + '_moves/'
    os.makedirs(move_image_dir, exist_ok=True)
    sheet_image_path = move_image_dir + ''
    move_image_path_list = []
    for move_idx in range(move_num_on_sheet):
        x1, y1, x2, y2 = move_rect_list[move_idx]
        img_cropped = sheet_image[y1:y2, x1:x2]

        color = "white" if move_idx % 2 == 0 else "black"
        move_name = str((move_idx // 2) + 1) + "_" + color
        move_image_path = move_image_dir + '/' + move_name + '.png'
        move_image_path_list.append(move_image_path)
        cv2.imwrite(move_image_path, img_cropped)

    return move_image_path_list


def main(args):
    data_sheet = get_data_sheet(args)  # for each score sheet
    image_path_list, move_list = get_data_move(data_sheet, args)
    with open(args.output_move_label, "w") as f:
        for path, move in zip(image_path_list, move_list):
            f.write(f"{path} {move}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="data/HCS")
    parser.add_argument('--output_move_label', default="output_move_label.txt")
    parser.add_argument('--output_image_dir', default="move_dir/")
    parser.add_argument('--plt_show', '-p', action='store_true')
    args = parser.parse_args()

    main(args)
