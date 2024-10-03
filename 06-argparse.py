# (zgp_efficientpy38t113) wangwei83@wangwei83-System-Product-Name:~/Desktop/python-test$ python 06-argparse.py --name wangwei --age 40
# Hello, wangwei! You are 40 years old.
# (zgp_efficientpy38t113) wangwei83@wangwei83-System-Product-Name:~/Desktop/python-test$ 

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example")

# Add arguments
parser.add_argument('--name', type=str, help='Your name')
parser.add_argument('--age', type=int, help='Your age')

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"Hello, {args.name}! You are {args.age} years old.")

