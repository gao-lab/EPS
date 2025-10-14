import numpy as np

from expression_copilot.mi import calc_eps


def test_mi():
    for k in range(3, 5):
        print(f"Testing k={k}")
        image_emb = np.random.rand(100, 10).astype(np.float32)
        for nd in range(1, 3):
            print(f"Testing nd={nd}")
            gene_expr = np.random.rand(100, nd).astype(np.float32).T
            # remove the extra dim
            gene_expr = np.squeeze(gene_expr)
            eps = calc_eps(image_emb.copy(), gene_expr.copy(), k=k)
            print(f"EPS: {eps}")
            sps = eps.mean()
            print(f"SPS: {sps}")


if __name__ == "__main__":
    test_mi()
    print("All MI tests passed.")
