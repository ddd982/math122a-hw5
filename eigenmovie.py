import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from glob import glob

# Identify naming pattern of files to be read into Python
dir = '.'
jpg_files = sorted(glob(f'{dir}/1970s_Version/*.jpg'))

# Compute number of image files
M = len(jpg_files)

# Compute size of image files
img = io.imread(jpg_files[0], as_gray=True)
m, n = img.shape

# run movie
ax = None
for i in range(0, M):
    filename = jpg_files[i]
    img = io.imread(filename, as_gray=True)
    if ax is None:
        ax = plt.imshow(img, cmap=plt.cm.gray)
    else:
        ax.set_data(img)
    plt.pause(.001)
    plt.draw()

# ---------------- (a) embed frames as columns ----------------
trainingDorms = np.zeros((m*n, M))
for i in range(0, M):
    filename = jpg_files[i]
    img = io.imread(filename, as_gray=True)
    #-------- your code begins here--
    trainingDorms[:, i] = img.flatten()
    #-------- your code ends here----

avgDorm = np.mean(trainingDorms, axis=1)[:, np.newaxis]

#-------- your code begins here-------
img = np.reshape(avgDorm, (m, n))
#-------- your code ends here----
plt.imshow(img, cmap=plt.cm.gray)
plt.title("mean frame")
plt.show()

# Compute eigendorms on mean-subtracted training data (reduced SVD)
X = trainingDorms - np.tile(avgDorm, (1, M))
#-------- your code begins here--
U, S, VT = np.linalg.svd(X, full_matrices=0)
#-------- your code ends here--

# (a) first four left singular vectors as images
fig, axs = plt.subplots(1, 4)
for k in range(4):
    axs[k].imshow(np.reshape(U[:, k], (m, n)), cmap=plt.cm.gray)
    axs[k].axis('off'); axs[k].set_title(f"u_{k+1}")
plt.tight_layout(); plt.show()

# ---------------- (b) 10th and 29th left singular vectors ----------------
fig, axs = plt.subplots(1, 2)
for a, k in zip(axs, [9, 28]):
    a.imshow(np.reshape(U[:, k], (m, n)), cmap=plt.cm.gray)
    a.axis('off'); a.set_title(f"u_{k+1}")
plt.tight_layout(); plt.show()
for k in [9, 28]:
    idx = np.argsort(-np.abs(VT[k, :]))[:3]
    print(f"u_{k+1}: frames with largest |v_{k+1}(t)|:", idx + 1)

# ---------------- (c) singular values on a log scale ----------------
plt.semilogy(np.arange(1, M+1), S, '.-')
plt.xlabel("i"); plt.ylabel(r"$\sigma_i$"); plt.title("singular values"); plt.grid(True, which="both", alpha=.3)
plt.show()
energy = np.cumsum(S**2) / np.sum(S**2)
for k in [1, 2, 3, 5, 10, 20, 50]:
    print(f"k={k:3d}  cumulative energy = {energy[k-1]:.4f}")

# ---------------- (d) first three right singular vectors vs time ----------------
t = np.linspace(3, 11, M)          # frames span 3am to 11am
for k in range(3):
    plt.plot(t, VT[k, :], label=rf"$v_{k+1}$")
plt.xlabel("time of day (hours, 3am to 11am)"); plt.ylabel("component value")
plt.title("first three right singular vectors vs time"); plt.legend(); plt.grid(alpha=.3)
plt.show()

# ---------------- (e) low-rank approximation of the movie ----------------
def lowrank_movie(k):
    """Rank-k approximation of the whole movie: columns are frames (pixels x M)."""
    Xk = U[:, :k] @ (S[:k, np.newaxis] * VT[:k, :])
    return Xk + avgDorm

def play(k):
    """Display the rank-k movie, k = 1 ... M."""
    frames = lowrank_movie(k)
    ax = None
    for i in range(M):
        img = np.reshape(frames[:, i], (m, n))
        if ax is None:
            ax = plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
        else:
            ax.set_data(img)
        plt.title(f"rank {k}, frame {i+1}")
        plt.pause(.001); plt.draw()

for k in [1, 5, 10, 20]:
    Xk = U[:, :k] @ (S[:k, np.newaxis] * VT[:k, :])
    print(f"rank {k}: relative error {np.linalg.norm(X - Xk)/np.linalg.norm(X):.4f}")

play(10)
