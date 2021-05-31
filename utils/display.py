import matplotlib.pyplot as plt
import matplotlib as mpl


def dispP(pa, pv, pav, gt):
    # display A, V, GT probability
    # dispP(a_prob, v_prob, output, target) - train
    #  -evaluation
    if pa is not None:
        plt.subplot(3, 2, 1)
        plt.imshow(pa.data.numpy())
        plt.title("A prob")

    if pv is not None:
        plt.subplot(3, 2, 2)
        plt.imshow(pv.data.numpy())
        plt.title("V prob")

    if gt is not None:
        plt.subplot(3, 2, 3)
        plt.imshow(gt.data.numpy())
        plt.title("GT")

    if pav is not None:
        plt.subplot(3, 2, 4)
        plt.imshow(pav.data.numpy())
        plt.title("AV prob")

    if pa is not None:
        plt.subplot(3, 2, 5)
        plt.imshow(gt.data.numpy() * pa.data.numpy())
        plt.title("A prob*GT")

    if pv is not None:
        plt.subplot(3, 2, 6)
        plt.imshow(gt.data.numpy() * pv.data.numpy())
        plt.title("V prob*GT")
        plt.show()



def disptest(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # disptest(Pa.transpose(), Pv.transpose(), Pav.transpose(), GT_a, GT_v, GT_av):
    SO_a, SO_v, SO_av, GT_a, GT_v, GT_av = SO_a.transpose(), SO_v.transpose(), \
                                           SO_av.transpose(), GT_a.transpose(),\
                                           GT_v.transpose(), GT_av.transpose()
    if SO_a is not None:
        plt.subplot(3, 2, 1)
        plt.imshow(SO_a)
        plt.title("A prob")

    if GT_a is not None:
        plt.subplot(3, 2, 2)
        plt.imshow(GT_a)
        plt.title("GT A prob")

    if SO_v is not None:
        plt.subplot(3, 2, 3)
        plt.imshow(SO_v)
        plt.title("V prob")

    if GT_v is not None:
        plt.subplot(3, 2, 4)
        plt.imshow(GT_v)
        plt.title("GT V prob")

    if SO_av is not None:
        plt.subplot(3, 2, 5)
        plt.imshow(SO_av)
        plt.title("AV prob")

    if GT_av is not None:
        plt.subplot(3, 2, 6)
        plt.imshow(GT_av)
        plt.title("GT AV prob")

    plt.show()