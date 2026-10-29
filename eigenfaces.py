import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from glob import glob

# ---------------- (a) read the 25 training images, build A = trainingFaces ----------------
dir = '.'
pgm_files = glob(f'{dir}/facedata/training/*.pgm')

# Compute number of image files
M = len(pgm_files)

# Compute size of image files
img = io.imread(pgm_files[0], as_gray=True)
m, n = img.shape

# Initialize matrix whose columns correspond to individual images
trainingFaces = np.zeros((m*n, M))

# Initialize matrix whose tiles correspond to individual images
allPersons = np.zeros((m*int(np.sqrt(M)), n*int(np.sqrt(M))))
count = 0

# Open figure window where you will display the images
fig1 = plt.figure()

# Loop over all image files
for i in range(0, M):

    # Acquire the filename of image i
    filename = pgm_files[i]

    # Read image i into Python as a 64 x 64 matrix
    img = io.imread(filename, as_gray=True)

    trainingFaces[:, i] = img.flatten()

    j = int(count % np.sqrt(M))
    k = int((count-count % np.sqrt(M))/np.sqrt(M))
    allPersons[k*m:(k+1)*m, j*n:(j+1)*n] = img
    count += 1

plt.imshow(allPersons, cmap=plt.cm.gray)
plt.title("(a) training faces")
plt.show()

# ---------------- (b) mean face ----------------
#-------- your code begins here--
avgFace = np.mean(trainingFaces, axis=1)[:, np.newaxis]
#-------- your code ends here----

plt.imshow(np.reshape(avgFace, (m, n)), cmap=plt.cm.gray)
plt.title("(b) mean face")
plt.show()

# ---------------- (c) eigenfaces on mean-subtracted training data ----------------
X = trainingFaces - np.tile(avgFace, (1, M))

#-------- your code begins here--
U, S, VT = np.linalg.svd(X, full_matrices=False)
#-------- your code ends here--

# Initialize matrix whose tiles correspond to eigenFaces
eigenFaces = np.zeros((m*int(np.sqrt(M)), n*int(np.sqrt(M))))
count = 0

# Loop over all image files
for i in range(0, M):

    # Take ith left singular vector
    #-------- your code begins here--
    img = U[:, i]
    #-------- your code ends here----

    j = int(count % np.sqrt(M))
    k = int((count-count % np.sqrt(M))/np.sqrt(M))
    eigenFaces[k*m:(k+1)*m, j*n:(j+1)*n] = np.reshape(img, (m, n))
    count += 1

plt.imshow(eigenFaces, cmap=plt.cm.gray)
plt.title("(c) eigenfaces")
plt.show()

# ---------------- (d) coordinates of each training face in the eigenface basis ----------------
W = U.T @ X          # column i = coefficients of training face i
print("(d) W shape:", W.shape)
print("(d) exact reconstruction:", np.allclose(U @ W + avgFace, trainingFaces))

names = [p.split('/')[-1] for p in pgm_files]

# ---------------- (e) in-sample recognition: Jackie Chan ----------------
test = io.imread(f'{dir}/facedata/training/Jackie_Chan_0006.pgm', as_gray=True)
f = test.flatten()[:, np.newaxis]
w = U.T @ (f - avgFace)                     # coordinates in eigenface basis
dist = np.linalg.norm(W - w, axis=0)        # Euclidean distance to each training face
best = np.argmin(dist)
print("(e) best match for Jackie Chan:", names[best], " distance:", round(float(dist[best]), 2))

# ---------------- (f) out-of-sample recognition: Clint Eastwood ----------------
clint = io.imread(f'{dir}/facedata/test/Clint_Eastwood_0005.pgm', as_gray=True)
f = clint.flatten()[:, np.newaxis]
w = U.T @ (f - avgFace)
dist = np.linalg.norm(W - w, axis=0)
best = np.argmin(dist)
recon = U @ w + avgFace                     # eigenface representation of Clint Eastwood
print("(f) best match for Clint Eastwood:", names[best], " distance:", round(float(dist[best]), 2))
print("(f) ||f - recon||:", round(float(np.linalg.norm(f - recon)), 2))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(clint, cmap=plt.cm.gray); ax[0].set_title("Clint Eastwood (input)")
ax[1].imshow(np.reshape(recon, (m, n)), cmap=plt.cm.gray); ax[1].set_title("eigenface representation")
ax[2].imshow(np.reshape(trainingFaces[:, best], (m, n)), cmap=plt.cm.gray); ax[2].set_title("best match: " + names[best].split('_0')[0])
for a in ax:
    a.axis("off")
plt.show()
