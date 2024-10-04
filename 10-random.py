import random

# Generate a random float between 0.0 and 1.0
random_float = random.random()
print(f"Random Float: {random_float}")


import random

# Generate a random integer between 1 and 10 (inclusive)
random_integer = random.randint(1, 10)
print(f"Random Integer: {random_integer}")

import random

# Choose a random element from a list
elements = ['apple', 'banana', 'cherry']
random_element = random.choice(elements)
print(f"Random Element: {random_element}")


import random

# Shuffle the elements of a list
elements = [1, 2, 3, 4, 5]
random.shuffle(elements)
print(f"Shuffled List: {elements}")


import random

# Generate a random sample of 3 elements from a list
elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random_sample = random.sample(elements, 3)
print(f"Random Sample: {random_sample}")


import random

# Generate a random float between 1.5 and 10.5
random_float_range = random.uniform(1.5, 10.5)
print(f"Random Float in Range: {random_float_range}")
