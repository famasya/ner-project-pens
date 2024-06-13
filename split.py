import os
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def datasplit(data: List[str], train_size: float, val_size: float) -> Tuple[List[str], List[str], List[str]]:
    train, val_test = train_test_split(data, train_size=train_size)
    val, test = train_test_split(val_test, test_size=val_size / (1 - train_size))
    return train, val, test

def main():
    with open("./dataset/dt1500_normalized.conll", "r") as file:
        contents = file.read()
    data = [content.replace(" -X- _ ", "\t") for content in contents.split("\n\n")]

    # split to train validation test
    train, valid, test = datasplit(data, 0.8, 0.1)
    current_time = datetime.now().isoformat()

    os.makedirs(f"splits/{current_time}", exist_ok=True)
    with open(f"splits/{current_time}/train.txt", "w") as file:
        file.write("\n\n".join(train))
    with open(f"splits/{current_time}/valid.txt", "w") as file:
        file.write("\n\n".join(valid))
    with open(f"splits/{current_time}/test.txt", "w") as file:
        file.write("\n\n".join(test))

    print(f"File written at splits/{current_time}")

if __name__ == "__main__":
    from datetime import datetime
    main()
