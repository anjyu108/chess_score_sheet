from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import tensorflow.keras.layers.experimental.preprocessing

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import argparse

# TODO introduce logger


# TODO fix hardcoded character list
characters_list = ['#', '+', '=',
                   '-', 'O',
                   'x',
                   '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                   'B', 'K', 'N', 'P', 'Q', 'R',
                   ]
max_len = 6

# Split Dataset into 1) train-90% 2) validation-5% 3) test-5%
train_ratio = {
        "train": 0.9,
        "validation": 0.05,
        "test": 0.05,
        }
assert sum(train_ratio.values()) == 1


# TODO Add argparse options to change default values
class ModelParams():
    def __init__(self):
        self.batch_size = 64
        self.padding_token = 99
        self.image_width = 256
        self.image_height = 64
global model_params
model_params = ModelParams()


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def get_dataset_ncs(args):
    """
    Load NCS dataset
    Args:
        args: input option with argparser
    """
    all_line = open(args.input_label, "r").readlines()
    move_list = [line.strip() for line in all_line]

    image_path_list = sorted(os.listdir(args.input_pred))
    image_path_list = [args.input_pred + '/' + path for path in image_path_list]
    image_path_list = image_path_list[:len(move_list)]  # remove empty moves


    words_list = [i for i in range(len(move_list))]
    print(f"before shuffle words_list: {words_list}")
    np.random.shuffle(words_list)
    print(f"after shuffle words_list: {words_list}")
    split_idx = int(train_ratio["train"] * len(words_list))
    train_samples = words_list[:split_idx]
    test_samples = words_list[split_idx:]

    validation_ratio = train_ratio["validation"] / (1 - train_ratio["train"])
    val_split_idx = int(validation_ratio * len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    # Print splitting results
    print(f"Number of words_list: {len(words_list)}")
    print(f"Total training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")

    def get_image_paths_and_labels(samples):
        paths = []
        corrected_samples = []
        for i in samples:
            paths.append(image_path_list[i])
            corrected_samples.append(move_list[i])

        return paths, corrected_samples

    train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
    test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

    ret = {
        "train":      {"img_paths": train_img_paths,
                       "labels": train_labels},
        "validation": {"img_paths": validation_img_paths,
                       "labels": validation_labels},
        "test":       {"img_paths": test_img_paths,
                       "labels": test_labels},
           }

    return ret


def distortion_free_resize(image):
    w, h = model_params.image_width, model_params.image_height
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]],
                   constant_values=model_params.padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(model_params.batch_size).cache().prefetch(tf.data.AUTOTUNE)


def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [layer for layer in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = keras.models.Model(input=layers[0].input, output=x)
    return new_model


def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = keras.models.Model(input=layers[0].input, output=x)
    return new_model


def main(args):
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    ######################
    # Dataset preprocess #
    ######################
    dataset_map = get_dataset_ncs(args)

    print("characters:", characters_list)
    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters_list))

    # Mapping characters to integers.
    global char_to_num
    char_to_num = StringLookup(vocabulary=characters_list, mask_token=None)

    # Mapping integers back to original characters.
    global num_to_char
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    train_ds = prepare_dataset(dataset_map["train"]["img_paths"],
                               dataset_map["train"]["labels"])
    validation_ds = prepare_dataset(dataset_map["validation"]["img_paths"],
                                    dataset_map["validation"]["labels"])
    test_ds = prepare_dataset(dataset_map["test"]["img_paths"],
                              dataset_map["test"]["labels"])

    # print dataset samples
    for data in train_ds.take(1):
        images, labels = data["image"], data["label"]

        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        for i in range(16):
            img = images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            # Gather indices where label!= padding_token.
            label = labels[i]
            indices = tf.gather(label, tf.where(
                tf.math.not_equal(label, model_params.padding_token)))
            # Convert to string.
            label = tf.strings.reduce_join(num_to_char(indices))
            label = label.numpy().decode("utf-8")

            title = f"{label}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

    if not args.not_plt_save:
        plt.savefig(os.path.join(args.output, 'input_samples.png'))

    ################
    # Define model #
    ################
    model = keras.saving.load_model(args.pretrained_model)

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    prediction_model.summary()

    ############
    # Training #
    ############
    # Define current accuracy value to print in the progress log
    validation_images = []
    validation_labels = []
    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

    def calculate_edit_distance(labels, predictions):
        # Get a single batch and convert its labels to sparse tensors.
        saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

        # Make predictions and convert them to sparse tensors.
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
        predictions_decoded = keras.backend.ctc_decode(
            predictions, input_length=input_len, greedy=True
        )[0][0][:, :max_len]
        sparse_predictions = tf.cast(
            tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
        )

        # Compute individual edit distances and average them out.
        edit_distances = tf.edit_distance(
            sparse_predictions, saprse_labels, normalize=False
        )
        return tf.reduce_mean(edit_distances)

    class EditDistanceCallback(keras.callbacks.Callback):
        def __init__(self, pred_model):
            super().__init__()
            self.prediction_model = pred_model

        def on_epoch_end(self, epoch, logs=None):
            edit_distances = []

            for i in range(len(validation_images)):
                labels = validation_labels[i]
                predictions = self.prediction_model.predict(validation_images[i])
                edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

            print(
                f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
            )

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    edit_distance_callback = EditDistanceCallback(prediction_model)
    if args.train_model:
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=args.epoch_num,
            callbacks=[edit_distance_callback],
        )

        model.save(args.output)

    ########
    # Test #
    ########

    print("Start testing")

    os.makedirs(args.output, exist_ok=True)

    # A utility function to decode the output of the network.
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        ctc_decode_ret = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
        results = ctc_decode_ret[0][0][:, :max_len]
        results_confidence = ctc_decode_ret[1]
        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text, results_confidence

    #  Let's check results on some test samples.
    img_paths = dataset_map["test"]["img_paths"]
    labels = dataset_map["test"]["labels"]
    test_ds = prepare_dataset(img_paths, labels)

    f = open(os.path.join(args.output, "ncs_predict_result.txt"), 'w')
    f.write("img_path,prediction_unconfidence,predicted_text\n")
    for batch_idx, batch in enumerate(test_ds):
        batch_images = batch["image"]
        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        preds = prediction_model.predict(batch_images)
        pred_texts, pred_confidence = decode_batch_predictions(preds)

        # FIXME: hardcoded batch_size=64
        for i in range(64):
            if i >= len(preds):
                break

            img_path_idx = 64 * batch_idx + i
            f.write(f"{img_paths[img_path_idx]},{pred_confidence[i][0]},{pred_texts[i]}\n")


        for i in range(16):
            if i >= len(preds):
                break

            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Pred: {pred_texts[i]} (unconf: {pred_confidence[i][0]:.3})"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

        if batch_idx == 0:
            plt.savefig(os.path.join(args.output, "ncs_predict_result.png"))
    f.close()

    print("End testing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pred', default="data/kanagawa_champ_2023/images_move/1/")
    parser.add_argument('--input_label', default="data/kanagawa_champ_2023/label/1.txt")
    parser.add_argument('--output', default="output_ncs")
    parser.add_argument('--pretrained_model', default="output_model_hcs/")
    parser.add_argument('--random_seed', '-r', type=int, default=None)
    parser.add_argument('--train_model', action='store_true')
    parser.add_argument('--epoch_num', '-e', type=int, default=50,
                        help='Shold be at least 50 for good accuracy')
    parser.add_argument('--not_plt_save', '-p', action='store_true')
    args = parser.parse_args()

    main(args)
