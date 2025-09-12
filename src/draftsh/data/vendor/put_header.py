from pathlib import Path
b = [i for i in Path(".").walk()]
for ff in b[0][2]:
    with open(ff, "r") as file:
        lines = file.readlines()
    with open(ff, "w") as file:
        for line_idx, line in enumerate(lines):
            new_line = line.replace(r"src\\draftsh\\data\\miscs\\vendor\\mastml\\LICENSE", r"src\\draftsh\\data\\vendor\\mastml\\LICENSE")
            file.write(new_line)