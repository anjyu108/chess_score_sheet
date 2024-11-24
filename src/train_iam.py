from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import tensorflow.keras.layers.experimental.preprocessing

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import argparse

# TODO introduce logger

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


def build_model():
    # Inputs to the model
    input_img = keras.Input(shape=(model_params.image_width, model_params.image_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # 1st conv block.
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # 2nd conv block.
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # 3rd conv block.
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv3",
    )(x)
    x = keras.layers.MaxPooling2D((1, 2), name="pool3")(x)

    x = keras.layers.Reshape(target_shape=(64, 4*512), name="reshape")(x)
    x = keras.layers.Dense(512, activation="relu", name="dense1")(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    x = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer_iam"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model


def get_dataset(args):
    """
    Load IAM dataset
    Args:
        args: input option with argparser
    """
    words_list = []

    words = open(f"{args.input}/words.txt", "r").readlines()
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] == "err":
            continue  # Don't use errored entries.
        words_list.append(line)

    np.random.shuffle(words_list)

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

    # Create the list of image path and labels (w/ all the info)

    base_image_path = os.path.join(args.input, "words")

    def get_image_paths_and_labels(samples):
        paths = []
        corrected_samples = []
        for (i, file_line) in enumerate(samples):
            line_split = file_line.strip()
            line_split = line_split.split(" ")

            # Each line split will have this format for the corresponding image:
            # part1/part1-part2/part1-part2-part3.png
            image_name = line_split[0]
            partI = image_name.split("-")[0]
            partII = image_name.split("-")[1]
            img_path = os.path.join(
                base_image_path, partI, partI + "-" + partII, image_name + ".png"
            )
            if os.path.getsize(img_path):
                paths.append(img_path)
                corrected_samples.append(file_line.split("\n")[0])

        return paths, corrected_samples

    train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
    test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

    # Create the list of `clean` labels (w/ only the transcription part of the label )
    train_labels_cleaned = []
    for label in train_labels:
        label = label.split(" ")[-1].strip()
        train_labels_cleaned.append(label)

    def clean_labels(labels):
        """
        `clean` means only the transcription part of the label
        """
        cleaned_labels = []
        for label in labels:
            label = label.split(" ")[-1].strip()
            cleaned_labels.append(label)
        return cleaned_labels

    validation_labels_cleaned = clean_labels(validation_labels)
    test_labels_cleaned = clean_labels(test_labels)

    ret = {
        "train":      {"img_paths": train_img_paths,
                       "labels": train_labels_cleaned},
        "validation": {"img_paths": validation_img_paths,
                       "labels": validation_labels_cleaned},
        "test":       {"img_paths": test_img_paths,
                       "labels": test_labels_cleaned},
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


def main(args):
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    ######################
    # Dataset preprocess #
    ######################
    dataset_map = get_dataset(args)

    # Build the character vocabulary
    global max_len  # FIXME global variable
    max_len = 0
    characters = set()

    for label in dataset_map["train"]["labels"]:
        max_len = max(max_len, len(label))
        for char in label:
            characters.add(char)

    characters = sorted(list(characters))

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))

    # Mapping characters to integers.
    global char_to_num
    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

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

            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")

    if args.plt_show:
        plt.show()
    if not args.no_plt_save:
        plt.savefig(os.path.join(args.output, 'input_samples.png'))

    ################
    # Define model #
    ################

    model = build_model()
    model.summary()

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

    # Run training
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

    # A utility function to decode the output of the network.
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :max_len
        ]
        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    #  Let's check results on some test samples.
    for batch in test_ds.take(1):
        batch_images = batch["image"]
        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        for i in range(16):
            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

    if args.plt_show:
        plt.show()
    if not args.no_plt_save:
        plt.savefig(os.path.join(args.output, 'predict_samples.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default="data/IAM")
    parser.add_argument('--output', '-o', default="output_model/")
    parser.add_argument('--random_seed', '-r', type=int, default=None)
    parser.add_argument('--epoch_num', '-e', type=int, default=50,
                        help='Shold be at least 50 for good accuracy')
    parser.add_argument('--plt_show', '-p', action='store_true')
    parser.add_argument('--no_plt_save', '-p', action='store_true')
    args = parser.parse_args()

    main(args)
