import argparse
import os


def main(args):
    with open(args.target_pgn_path) as f:
        target_data = f.read().strip().split("\n\n")[-1].split()
        target_data = [x for x in target_data if "```" not in x]
    with open(args.answer_pgn_path) as f:
        answer_data = f.read().strip().split("\n\n")[-1].split()
        answer_data = [x for x in answer_data if "```" not in x]
    if args.debug:
        print(f"target_data: {target_data}")
        print(f"answer_data: {answer_data}")

    # Compare target and answer PGN data, and calculate accuracy
    total = 0
    correct = 0
    for target, answer in zip(target_data, answer_data):
        total += 1
        if target == answer:
            correct += 1
        else:
            if args.debug:
                print(f"{total:3}: (target vs answer) {target} vs {answer}")
    # treat missing moves as incorrect
    # when target has more moves than answer, simply ignore the extra moves
    if len(answer_data) > len(target_data):
        total += len(answer_data) - len(target_data)

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%  ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_pgn_path", default="gpt-4o/pgn/img37.jpg.pgn")
    parser.add_argument("--answer_pgn_path", default="ocr-test-answer/img37.jpg.pgn")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
