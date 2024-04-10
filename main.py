import argparse
from tf_cifar10 import *
parser = argparse.ArgumentParser(
    prog='Machine Learning Benchmarks',
    description='A benchmark for the performance of a machine learning platform'
)


def main():
    train_cifar(use_all_gpus=False)
    print("Trying distributed strategy")
    train_cifar(use_all_gpus=True)


if __name__ == "__main__":
    main()