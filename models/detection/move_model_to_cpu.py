import sys
import torch


def main(gpu_model_path, cpu_model_path):

    # 1. load trained model
    model = torch.load(gpu_model_path)

    # 2. move model to CPU
    model.to("cpu")

    # 3. save model
    torch.save(model, cpu_model_path)

if __name__ == "__main__":

    gpu_model_path = sys.argv[1]
    cpu_model_path = sys.argv[2]

    main(gpu_model_path, cpu_model_path)