import ujson
import random


def sample_minimval(val_path, save_path, num_samples=1000):
    with open(val_path, "r") as f:
        val = ujson.load(f)
    val = random.sample(val, num_samples)
    with open(save_path, "w") as f:
        ujson.dump(val, f)


if __name__ == "__main__":
    val_path = "/home/data/datasets/moma_qa/val.json"
    save_path = "/home/data/datasets/moma_qa/minival.json"
    sample_minimval(val_path, save_path, num_samples=1000)