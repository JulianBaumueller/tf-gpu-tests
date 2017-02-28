# Init script for partial scripts

import _thread

from datasets import mnist
from datasets import svhn
from datasets import cifar

print("TensorFlow demo with additional test data:")
print()
print("1 - MNIST (Set of Handwriting Digits)")
print("2 - SVHN (Set of Street View House Numbers")
print("10 - CIFAR-10 (Set of Real Life Images)")
print("100 - CIFAR-100 (Set of Real Life Images in 32x32 Pixels)")
print()

menuInput = input("Choose a dataset you want to test: ")
print()

def menu(x):
    return {
        '1': mnist.init(),
        '2': svhn.init(),
        '3': cifar.init()
    }

menu(menuInput)