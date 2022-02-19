import seaborn as sns
import numpy as np

def Fraction(y_true, probabilities):
    
    TNF, R84, PIC, P3K = np.empty([1 ,9]), np.empty([1, 9]), np.empty([1, 9]), np.empty([1, 9])
    FLA, CpG, FSL, LPS, UST = np.empty([1, 9]), np.empty([1, 9]), np.empty([1, 9]), np.empty([1, 9]), np.empty([1, 9])
    
    for i in range(len(y_true)):
        if (y_true[i] == 0.0):
            TNF = np.vstack([TNF, probabilities[i]])
        if (y_true[i] == 1.0):
            R84 = np.vstack([R84, probabilities[i]])
        if (y_true[i] == 2.0):
            PIC = np.vstack([PIC, probabilities[i]])
        if (y_true[i] == 3.0):
            P3K = np.vstack([P3K, probabilities[i]])
        if (y_true[i] == 4.0):
            FLA = np.vstack([FLA, probabilities[i]])
        if (y_true[i] == 5.0):
            CpG = np.vstack([CpG, probabilities[i]])
        if (y_true[i] == 6.0):
            FSL = np.vstack([FSL, probabilities[i]])
        if (y_true[i] == 7.0):
            LPS = np.vstack([LPS, probabilities[i]])
        if (y_true[i] == 8.0):
            UST = np.vstack([UST, probabilities[i]])
        if (i == len(y_true) - 1):
            TNF, R84, PIC, P3K, FLA = TNF[1:], R84[1:], PIC[1:], P3K[1:], FLA[1:]
            CpG, FSL, LPS, UST = CpG[1:], FSL[1:], LPS[1:], UST[1:]
            
    polarized_fraction = np.empty([1, 9])
    polarized_fraction = np.vstack([polarized_fraction, TNF])
    polarized_fraction = np.vstack([polarized_fraction, R84])
    polarized_fraction = np.vstack([polarized_fraction, PIC])
    polarized_fraction = np.vstack([polarized_fraction, P3K])
    polarized_fraction = np.vstack([polarized_fraction, FLA])
    polarized_fraction = np.vstack([polarized_fraction, CpG])
    polarized_fraction = np.vstack([polarized_fraction, FSL])
    polarized_fraction = np.vstack([polarized_fraction, LPS])
    polarized_fraction = np.vstack([polarized_fraction, UST])
    polarized_fraction = polarized_fraction[1:]
    
    return polarized_fraction
