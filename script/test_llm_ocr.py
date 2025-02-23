import argparse
import os
import base64
import datetime
from abc import abstractmethod, ABC
from dotenv import load_dotenv
from openai import OpenAI
import mimetypes
from google import genai
import anthropic


OPENAI_MODEL_LIST = ["gpt-4o", "gpt-4o-mini", "o3-mini"]
GOOGLE_MODEL_LIST = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]
ANTHROPIC_MODEL_LIST = ["claude-3-5-sonnet-20241022"]


def main(args):
    if args.model_name in OPENAI_MODEL_LIST:
        pgn_gen = PGN_GEN_OPENAI(
            args.environ_path, args.input_image_path + "_debug.log"
        )
    elif args.model_name in GOOGLE_MODEL_LIST:
        pgn_gen = PGN_GEN_GOOGLE(
            args.environ_path, args.input_image_path + "_debug.log"
        )
    elif args.model_name in ANTHROPIC_MODEL_LIST:
        pgn_gen = PGN_GEN_ANTHROPIC(
            args.environ_path, args.input_image_path + "_debug.log"
        )
    else:
        raise ValueError("Invalid model_name: " + args.model_name)

    pgn_data, input_token, output_token = pgn_gen.generate_pgn(
        args.model_name, args.input_image_path
    )

    with open(args.input_image_path + ".pgn", "w") as f:
        print(pgn_data, file=f)
    with open(args.input_image_path + "_input_token.txt", "w") as f:
        print(input_token, file=f)
    with open(args.input_image_path + "_output_token.txt", "w") as f:
        print(output_token, file=f)


class PGN_GEN(ABC):
    @abstractmethod
    def generate_pgn(self, model_name, input_image_path):
        """
        Generate PGN data from chess handwriting scoresheet image.

        Returns:
            str: PGN data representing the scoresheet.
            float: input token
            float: output token
        """
        pass


class PGN_GEN_OPENAI(PGN_GEN):
    def __init__(self, environ_path, debug_log_path):
        self.debug_log_f = open(debug_log_path, "w")
        load_dotenv(environ_path, override=True)
        self.OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
        self.INPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN = float(
            os.getenv("INPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN")
        )
        self.OUTPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN = float(
            os.getenv("OUTPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN")
        )
        self.client = OpenAI()

    def local_image_to_data_url(image_path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            print(
                "Warning: Use `octet-stream as image format because could not determine MIME type. Ensure the file has a valid image extension (e.g., .png, .jpg, .jpeg)."
            )
            mime_type = "application/octet-stream"

        # Read and encode the image file
        try:
            with open(image_path, "rb") as image_file:
                base64_encoded_data = base64.b64encode(image_file.read()).decode(
                    "utf-8"
                )
        except Exception as e:
            print(f"Error reading image file: {e}")
            return None

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def generate_pgn(self, model_name, input_image_path):
        print(f"Executed datetime: {datetime.datetime.today()}", file=self.debug_log_f)
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
        image_data_url = PGN_GEN_OPENAI.local_image_to_data_url(input_image_path)
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
                model=model_name,
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

        input_cost = (
            self.INPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN * tokens["input"] / 1_000_000
        )
        output_cost = (
            self.OUTPUT_TOKEN_COST_DOLLAR_PER_1M_TOKEN * tokens["output"] / 1_000_000
        )
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

        return pgn_data, tokens["input"], tokens["output"]


class PGN_GEN_GOOGLE(PGN_GEN):
    def __init__(self, environ_path, debug_log_path):
        self.debug_log_f = open(debug_log_path, "w")
        load_dotenv(environ_path, override=True)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_pgn(self, model_name, input_image_path):
        print(f"Executed datetime: {datetime.datetime.today()}", file=self.debug_log_f)
        prompt = "Generate PGN data from the input chess handwriting scoresheet image. The response contexnt should NOT include anything other than the PGN data."
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        print(response, file=self.debug_log_f)

        ret = (
            response.text,
            response.usage_metadata.prompt_token_count,
            response.usage_metadata.candidates_token_count,
        )
        return ret


class PGN_GEN_ANTHROPIC(PGN_GEN):
    def __init__(self, environ_path, debug_log_path):
        self.debug_log_f = open(debug_log_path, "w")
        load_dotenv(environ_path, override=True)
        self.client = anthropic.Anthropic()

    def get_base64_encoded_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
            base_64_encoded_data = base64.b64encode(binary_data)
            base64_string = base_64_encoded_data.decode("utf-8")
            return base64_string

    def generate_pgn(self, model_name, input_image_path):
        print(f"Executed datetime: {datetime.datetime.today()}", file=self.debug_log_f)
        client = anthropic.Anthropic()

        system_message = "You are a AI to generate PGN data from chess handwriting scoresheet image. The response contexnt should NOT include anything other than the PGN data."
        user_message = "Generate PGN of this image"
        image_data = PGN_GEN_ANTHROPIC.get_base64_encoded_image(input_image_path)

        message = client.messages.create(
            model=model_name,
            max_tokens=1000,
            temperature=0,
            system=system_message,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": user_message},
                    ],
                },
            ],
        )
        print(message, file=self.debug_log_f)

        # import pdb;pdb.set_trace()
        pgn_data = message.content[0].text
        tokens = {
            "input": message.usage.input_tokens,
            "output": message.usage.output_tokens,
        }


        return pgn_data, tokens["input"], tokens["output"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", default="ocr-test/img135.jpg")
    parser.add_argument("--environ_path", default=".env")
    all_model_list = OPENAI_MODEL_LIST + GOOGLE_MODEL_LIST + ANTHROPIC_MODEL_LIST
    parser.add_argument(
        "--model_name", type=str, choices=all_model_list, default="gpt-4o"
    )
    args = parser.parse_args()

    main(args)
