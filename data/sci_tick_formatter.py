
import numpy as np
# from matplotlib import pyplot as plt

# ref: https://stackoverflow.com/questions/20127388/scientific-notation-on-each-tick-in-the-default-font-in-pyplot
def SciFormatter(x, lim):
    if x <= 100 and x >= 0.01:
        return "${0:.1f}$".format(x)

    if x == 0:
        return "$0$"

    firstNum = "{0:.1f}".format(
        np.sign(x) * 10 ** (-np.floor(np.log10(abs(x))) + np.log10(abs(x)))
    )
    firstNum = "{" + firstNum + "}"

    exponent = "{0:.0f}".format(np.floor(np.log10(abs(x))))

    exponent = "{" + exponent + "}"

    # about math regular:
    # https://stackoverflow.com/questions/27698377/how-do-i-make-sans-serif-superscript-or-subscript-text-in-matplotlib
    return "$" + f"{firstNum}x10^{exponent}" + "$"