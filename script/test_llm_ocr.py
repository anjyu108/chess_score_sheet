import argparse
import os
import base64
import datetime
from abc import abstractmethod, ABC
from dotenv import load_dotenv
from openai import OpenAI
import mimetypes


def main(args):
    pgn_gen_openai = PGN_GEN_OPENAI(args.environ_path,
                                    args.input_image_path + "_debug.log")
    pgn_data, total_cost = pgn_gen_openai.generate_pgn(args.input_image_path)
    with open(args.input_image_path + ".pgn", "w") as f:
        print(pgn_data, file=f)
    with open(args.input_image_path + "_price.txt", "w") as f:
        print(total_cost, file=f)


class PGN_GEN(ABC):
    @abstractmethod
    def generate_pgn(self, input_image_path):
        """
        Generate PGN data from chess handwriting scoresheet image.

        Returns:
            str: PGN data representing the scoresheet.
            float: cost to generate the PGN data in dollar.
        """
        pass


class PGN_GEN_OPENAI(PGN_GEN):
    def __init__(self, environ_path, debug_log_path):
        self.debug_log_f = open(debug_log_path, "w")
        load_dotenv(environ_path, override=True)
        self.OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
        self.INPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN = float(os.getenv("INPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN"))
        self.OUTPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN = float(os.getenv("OUTPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN"))
        self.client = OpenAI()

    def local_image_to_data_url(image_path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            print("Warning: Use `octet-stream as image format because could not determine MIME type. Ensure the file has a valid image extension (e.g., .png, .jpg, .jpeg).")
            mime_type = "application/octet-stream"

        # Read and encode the image file
        try:
            with open(image_path, "rb") as image_file:
                base64_encoded_data = base64.b64encode(image_file.read()
                                                       ).decode(
                    "utf-8"
                )
        except Exception as e:
            print(f"Error reading image file: {e}")
            return None

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def generate_pgn(self, input_image_path):
        print(f"Executed datetime: {datetime.datetime.today()}",
              file=self.debug_log_f)
        messages = []

        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a AI to generate PGN data from chess handwriting scoresheet image. The response contexnt should NOT include anything other than the PGN data.",
                    }
                ],
            }
        )

        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": "Generate PGN of this image"}],
        }
        image_data_url = \
            PGN_GEN_OPENAI.local_image_to_data_url(input_image_path)
        user_message["content"].append(
            {"type": "image_url", "image_url": {"url": image_data_url}}
        )
        messages.append(user_message)

        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "This is the PGN format representing your scoresheet.",
                    }
                ],
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=self.OPENAI_DEPLOYMENT_NAME,
                messages=messages,
            )
        except Exception as e:
            print(f"Error: {e}")
            return

        print(response, file=self.debug_log_f)
        pgn_data = response.choices[0].message.content
        tokens = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        }

        input_cost = self.INPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN * tokens["input"] / 1_000_000
        output_cost = self.OUTPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN * tokens["output"] / 1_000_000
        print(
            f"input:  {input_cost:.6f} $, {tokens["input"]:5} tokens",
            file=self.debug_log_f,
        )
        print(
            f"output: {output_cost:.6f} $, {tokens["output"]:5} tokens",
            file=self.debug_log_f,
        )
        print(
            f"total: {input_cost + output_cost:.6f} $",
            file=self.debug_log_f,
        )

        return pgn_data, input_cost + output_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", default="ocr-test/img135.jpg")
    parser.add_argument("--environ_path", default=".env")
    args = parser.parse_args()

    main(args)
