import numpy as np

# Load the .npz file
npz_file = np.load(r'D:\FIT\Senior Year\FALL 2024\MLA\MLA Midterm\starter_code\data\train_sparse.npz')

# List all the arrays stored in the .npz file
print("Keys in the .npz file:", npz_file.files)

# Access and print each array
for key in npz_file.files:
    print(f"{key}: {npz_file[key]}")

# Don't forget to close the file if you're done
npz_file.close()
