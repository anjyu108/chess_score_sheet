import cv2
import numpy as np
import random

def main():
    img = cv2.imread('data/sample_score_sheet.jpg', -1)
    img = cv2.GaussianBlur(img, (3,3), 0) # ガウシアンフィルタ

    # clip paper
    x_size = 2000
    y_size = int(x_size * 1.412)  # Assume the paper is B5 size
    img = clipped_paper(img, x_size, y_size)
    cv2.imwrite("data/clipped_paper.png", img)

    # recognize notations
    notations = list_notations(img)
    print('notations:', notations)

    # cv2.imshow('window', img)
    # cv2.waitKey(5000)


def clipped_paper(img, x_size, y_size):
    """
    Return image of the paper, which is clipped from the argument image

    Args:
        img: source image
        x_size: output image x size
        y_size: output image y size
    """

    # Detect corners of the paper
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_th1 = cv2.threshold(img_gray, 220, 255, cv2.THRESH_TOZERO_INV)
    img_not = cv2.bitwise_not(img_th1)
    ret, img_th2 = cv2.threshold(img_not, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_th2, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find paper contour
    picture_outline = {'contour': None, 'size': 0}  # should be max size
    paper_outline = {'contour': None, 'size': 0}  # should be second max size
    for con in contours:
        size = cv2.contourArea(con)
        if size >= picture_outline['size']:
            picture_outline['contour'] = con
            picture_outline['size'] = size
            continue
        if size >= paper_outline['size']:
            paper_outline['contour'] = con
            paper_outline['size'] = size
            continue

    # reshape paper contour to the original size ratio
    epsilon = 0.1*cv2.arcLength(paper_outline['contour'], True)
    paper_corners = cv2.approxPolyDP(paper_outline['contour'], epsilon, True)
    reshaped_con = np.array([[[0, 0]], [[x_size, 0]],
                            [[x_size, y_size]], [[0, y_size]]],
                            dtype="int32")
    reshape_M = cv2.getPerspectiveTransform(np.float32(paper_corners),
                                            np.float32(reshaped_con))
    img_trans = cv2.warpPerspective(img, reshape_M, (x_size, y_size))

    return img_trans


def list_notations(img):
    # detect table corner
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(img_gray, 200, 200)  # TODO fix magic number

    # thick borderline
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    edge = cv2.dilate(edge, kernel)

    # detected edge in the paper
    cv2.imwrite("data/edge.png", edge)

    # clip notation table part
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    curves = []
    notation_table_contour = None
    for contour, hierarchy in zip(contours, hierarchy[0]):
        curve = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        if len(curve) != 4:
            continue
        size = cv2.contourArea(contour)
        if size <= 1000000:  # TODO fix magic number
            continue

        curves.append(curve)
        notation_table_contour = contour
    assert(notation_table_contour is not None)

    # draw rect to show the notation table part in the paper
    rect_image = img.copy()
    for i, curve in enumerate(curves):
        p1, p3 = curve[0][0], curve[2][0]
        x1, y1, x2, y2 = p1[0], p1[1], p3[0], p3[1]
        r, g, b = random.random()*255, random.random()*255, random.random()*255
        cv2.rectangle(rect_image, (x1, y1), (x2, y2), (b, g, r), thickness=2)
    cv2.imwrite("data/notation_table_part_rect.png", rect_image)

    # clip notation table part
    x,y,w,h = cv2.boundingRect(notation_table_contour)  # FIXME comprehensive var name
    img_notation_table = img[y:y+h, x:x+w]
    cv2.imwrite("data/notation_table.png", img_notation_table)

    # TODO remove this part
    h, w, _ = img_notation_table.shape
    print('width:  ', w, 'height: ', h)

    img_notation_table_rec = img_notation_table.copy()
    column_move_num = 30  # 1 column has 30 moves
    elem_h = h / column_move_num
    buffer = 0.2
    row_elem_size_ratio = [1, 10, 10] * 2  # [header, note, note] * 2
    for row in range(column_move_num):
        # if row % 2 == 0:
        #     continue
        y1 = int(row*elem_h - elem_h*buffer)
        y2 = int((row+1)*elem_h + elem_h*buffer)
        for column in range(len(row_elem_size_ratio)):
            x1, x2 = 10, w-10
            x1 = int(sum(row_elem_size_ratio[:column]) / sum(row_elem_size_ratio) * w)
            x2 = int(sum(row_elem_size_ratio[:column+1]) / sum(row_elem_size_ratio) * w)

            # print(row+1, column+1, x1, x2, y1, y2)
            r, g, b = random.random()*255, random.random()*255, random.random()*255
            cv2.rectangle(img_notation_table_rec, (x1, y1), (x2, y2), (b, g, r), thickness=2)
    cv2.imwrite("data/notation_table_rect.jpg", img_notation_table_rec)

    return []



if __name__ == '__main__':
    main()
