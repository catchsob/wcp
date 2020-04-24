CHAR_VECTOR = "ABCDEFGHJKLMNPQRSTUVWXYZ012356789-" #enos

letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1

img_w, img_h = 128, 64

# Network parameters
batch_size = 192 #enos
val_batch_size = 192 #enos

downsample_factor = 4
max_text_len = 8 #enos