import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from graph_builder import get_skeleton_from_mask, build_graph_from_skeleton, simplify_graph


def create_synthetic_mask(shape=(256, 256)):
    img = np.zeros(shape, dtype=np.uint8)
    # horizontal main road
    cv2.line(img, (10, 128), (245, 128), color=255, thickness=6)
    # vertical branch
    cv2.line(img, (128, 10), (128, 128), color=255, thickness=6)
    # diagonal spur
    cv2.line(img, (180, 128), (230, 60), color=255, thickness=4)
    return img


def test_graph_pipeline():
    mask = create_synthetic_mask()

    sk = get_skeleton_from_mask(mask)

    G = build_graph_from_skeleton(sk)
    G = simplify_graph(G, min_length=2)

    print(f"Skeleton pixels: {sk.sum()}")
    print(f"Graph nodes: {len(G.nodes())}")
    print(f"Graph edges: {len(G.edges())}")

    # visualize
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(mask, cmap="gray")
    ax[0].set_title("Synthetic Thick Mask")
    ax[0].axis("off")

    ax[1].imshow(sk, cmap="gray")
    ys = [n[0] for n in G.nodes()]
    xs = [n[1] for n in G.nodes()]
    ax[1].scatter(xs, ys, c="red", s=30)
    for u, v, d in G.edges(data=True):
        pix = np.array(d["pixels"])
        ax[1].plot(pix[:, 1], pix[:, 0], c="yellow")
    ax[1].set_title("Skeleton + Graph")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_graph_pipeline()
