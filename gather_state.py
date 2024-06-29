import easyocr

reader = easyocr.Reader(["en"], gpu=True)


def get_gather_state(img_np):
    gather_state = reader.readtext(img_np)
    gather_state = [item[1] for item in gather_state]
    return " ".join(gather_state)
