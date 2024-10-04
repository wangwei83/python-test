import os

current_directory = os.getcwd()
print(f"Current Working Directory: {current_directory}")

import os

# Change to a specific directory
os.chdir('C:/')
print(f"Changed Working Directory: {os.getcwd()}")

import os

# List all files and directories in the current directory
files_and_directories = os.listdir()
print(f"Files and Directories: {files_and_directories}")


import os

# Create a new directory
os.mkdir('1_new_directory')
print("Directory '1_new_directory' created")

import os

# Remove an existing directory
os.rmdir('1_new_directory')
print("Directory '1_new_directory' removed")

import os

# Get the value of an environment variable
home_directory = os.getenv('HOME')
print(f"Home Directory: {home_directory}")

