import fitz
import os
import argparse


def main(args):
    pass
    pdf_file = fitz.open(args.input)

    images_list = []
    for page_num in range(len(pdf_file)):
        page_content = pdf_file[page_num]
        images_list.extend(page_content.get_images())

    if len(images_list) == 0:
        print(f"No images found in {args.input}")
        return -1

    os.makedirs(args.output, exist_ok=True)

    for i, image in enumerate(images_list, start=1):
        base_image = pdf_file.extract_image(image[0])

        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        image_name = str(i) + "." + image_ext
        with open(os.path.join(args.output, image_name) ,"wb") as image_file:
            image_file.write(image_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/kanagawa_champ_2023/data_pdf/R1.pdf")
    parser.add_argument("--output", default="data/kanagawa_champ_2023/images/")
    args = parser.parse_args()

    main(args)
