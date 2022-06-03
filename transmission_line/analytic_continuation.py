import numpy as np
from scipy.linalg import qr, svd


def vandermonde(m: int, s_list: np.ndarray) -> np.ndarray:
    n = len(s_list)
    mat = np.empty((n, m + 1))
    for i in range(n):
        for j in range(m + 1):
            mat[i, j] = s_list[i] ** j
    return mat


def rational_coefficients(
        frequencies: np.ndarray,
        simulated_values: np.ndarray,
        na: int = 20,
        reduce_orders: bool = True,
        nd: int = 15):
    nb = na + 1

    # build Vandermonde matrix of frequencies
    A = vandermonde(na, frequencies)

    # build matrix with simulated data and frequencies
    H = np.diag(simulated_values)
    B_f = vandermonde(nb, frequencies)
    B = H @ B_f

    if reduce_orders:
        diff_nb_na = nb - na

        # concatenate to large system matrix C = [A -B]
        C = np.concatenate((A, -B), axis=1)

        # singular value decomposition of C to find only singular values
        s = svd(C, compute_uv=False)

        # estimate rank of C by only considering singular values over a
        # threshold as nonzero
        rank = np.sum(s / np.max(s) >= (10 ** (-nd)))

        # redefine reduced polynomial orders
        na = int(np.floor((rank - diff_nb_na - 1) / 2))
        nb = na + diff_nb_na

        if nb < 0:
            raise ValueError(
                'Rank of system is too low to support na and nb, '
                'choose new coefficients.'
            )

    # reduce matrices to new orders
    A = A[:, : na + 1]
    B = B[:, : nb + 1]

    # QR decomposition of the frequency dependent part
    Q, R = qr(A)

    # slice R to get the part that is non-zero and coupled to A
    R11 = R[: na + 1, :]

    # compute matrix product of Q^T and -B
    QTB = Q.T @ -B

    # slice QTB in two parts that are coupled to B
    R12 = QTB[: na + 1, :]
    R22 = QTB[na + 1:, :]

    # singular value decomposition of R22
    _, _, Vh = svd(R22)

    # TLS step, last column of the hermitian transpose of Vh = V is b (scaled)
    b = np.conjugate(Vh).T[:, -1]
    # solve for a
    a = -np.linalg.inv(R11) @ R12 @ b

    return a, b


def y_in(a, b, w):
    na = len(a)
    nb = len(b)
    w_num = np.array([w ** k for k in range(na)])
    w_denom = np.array([w ** k for k in range(nb)])
    return a.dot(w_num) / b.dot(w_denom)
