import random
import json

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)

# Set the seed for reproducibility
random.seed(45)


# selected_numbers[:10], len(selected_numbers)  # Display the first 10 numbers and the total count


name = "swap"
# name = "swap_upper"
# name = "upper"

file_name = f"./factory/{name}_factory.json"
file_test =f"{name}_test.json"
file_train =f"{name}_train.json"

total = load_json(file_name)


# Create a range of numbers from 0 to len(total)-1 (len(total) total numbers)
numbers = range(len(total))

# Randomly select 30% of the numbers from the range
prop = 0.3
# prop = 0.5
selected_numbers = random.sample(numbers, int(prop * len(numbers)))


test_list = [total[i] for i in selected_numbers]
train_list = [num for num in total if num not in test_list]


save_json(test_list, file_test)
save_json(train_list, file_train)


