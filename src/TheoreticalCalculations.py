import numpy as np
from scipy.special import erfc, comb

class TheoreticalCalculations:
    @staticmethod
    def bpsk_coherent_theoretical_ber_rayleigh_fading(snr_linear):
        ber = 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))
        return ber

    @staticmethod
    def bpsk_coherent_theoretical_ber_awgn(snr_linear):
        ber = 0.5 * erfc(np.sqrt(snr_linear))
        return ber

    @staticmethod
    def bpsk_coherent_theoretical_ber_mrc(snr_linear, n_antennas):
        mu = np.sqrt(snr_linear / (1 + snr_linear))
        total_ber = 0
        for k in range(n_antennas):
            term = comb(n_antennas - 1 + k, k) * ((1 + mu) / 2) ** k
            total_ber += term
        return (0.5 * (1 - mu)) ** n_antennas * total_ber
