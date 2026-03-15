import pyautogui
import json
import time

# time.sleep(5)

def run_in_map(file_name):
    sequence = json.load(open(file_name, "r"))
    for location in sequence:
        print(location)
        x,y = location
        # x +=20
        # y +=20
        pyautogui.moveTo(x,y, 0.5, pyautogui.easeInOutQuad)
        time.sleep(0.4)
        pyautogui.rightClick()
        time.sleep(7)

def clean_duplicate_vector(file_name):
    sequence = json.load(open(file_name, "r"))
    print(len(sequence))
    for_delete = []
    for index, vector in enumerate(sequence):
        if index+1 == len(sequence):
            break
        if vector == sequence[index+1]:
            for_delete.append(index)
    print(for_delete)
    print(len(for_delete))
    for index in reversed(for_delete) :
        del sequence[index]
    print(len(sequence))
    str_sequence = json.dumps(sequence)
    with open(file_name, "w") as f:
        f.write(str_sequence)

folder_name = "map_to_lymhurst"
for number in range(13,15):
    time.sleep(15)
    file_name = f"{folder_name}/t{number}.json"
    run_in_map(file_name)