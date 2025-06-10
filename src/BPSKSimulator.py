from typing import Any

import numpy as np


class BPSKSimulation:
    @staticmethod
    def generate_random_bits(n_bits: int) -> np.ndarray:
        return np.where(np.random.rand(n_bits) > 0.5, 1, 0)

    @staticmethod
    def generate_gaussian_noise(n_bits: int, variance: float) -> np.ndarray:
        return np.sqrt(variance / 2) * np.random.randn(n_bits)

    @staticmethod
    def generate_complex_gaussian_noise(shape: tuple, variance: float) -> np.ndarray:
        real = np.sqrt(variance / 2) * np.random.randn(*shape)
        imag = np.sqrt(variance / 2) * np.random.randn(*shape)
        return real + 1j * imag

    @staticmethod
    def generate_rayleigh_fading(n_bits: int, scale: float = np.sqrt(1 / 2)) -> np.ndarray:
        return np.random.rayleigh(scale=scale, size=n_bits)

    @staticmethod
    def generate_complex_rayleigh_fading(shape: tuple) -> np.ndarray:
        # Complex Rayleigh fading with proper normalization
        real = np.random.randn(*shape) / np.sqrt(2)
        imag = np.random.randn(*shape) / np.sqrt(2)
        return real + 1j * imag

    @staticmethod
    def apply_coding(bits: np.ndarray) -> np.ndarray:
        return np.where(bits == 0, -1, 1)

    @staticmethod
    def apply_decoding(bits: np.ndarray) -> np.ndarray:
        return np.where(bits == 1, 1, 0)

    @staticmethod
    def thresholding(bits: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(bits > threshold, 1, 0)

    @staticmethod
    def thresholding_complex(complex_vals: np.ndarray) -> np.ndarray:
        return np.where(np.real(complex_vals) > 0, 1, 0)

    @staticmethod
    def apply_fading(bits: np.ndarray) -> np.ndarray:
        fading = BPSKSimulation.generate_rayleigh_fading(n_bits=bits.shape[0])
        return fading * bits

    @staticmethod
    def apply_awgn_channel(bits: np.ndarray, variance: float) -> np.ndarray:
        noise = BPSKSimulation.generate_gaussian_noise(bits.shape[0], variance)
        return bits + noise

    @staticmethod
    def calculate_error(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
        return np.mean(np.where(bits_a != bits_b, 1, 0))

    @staticmethod
    def simulate_alamouti(n_bits: int, variance: float, n_rx_antennas: int) -> tuple:
        """
        Alamouti STBC implementation for BPSK
        """
        # Ensure even number of bits
        if n_bits % 2 == 1:
            n_bits -= 1

        n_symbol_pairs = n_bits // 2

        # Generate bits and BPSK symbols
        random_bits = BPSKSimulation.generate_random_bits(n_bits)
        symbols = BPSKSimulation.apply_coding(random_bits)  # -1, +1

        # Group symbols in pairs
        s1_vec = symbols[0::2]  # s1, s3, s5, ...
        s2_vec = symbols[1::2]  # s2, s4, s6, ...

        # Generate channel matrix: h[rx_ant, tx_ant, symbol_pair]
        h = BPSKSimulation.generate_complex_rayleigh_fading((n_rx_antennas, 2, n_symbol_pairs))

        # Generate noise
        noise = BPSKSimulation.generate_complex_gaussian_noise((n_rx_antennas, 2, n_symbol_pairs), variance)

        # Alamouti transmission and reception
        r1 = np.zeros((n_rx_antennas, n_symbol_pairs), dtype=complex)
        r2 = np.zeros((n_rx_antennas, n_symbol_pairs), dtype=complex)

        for i in range(n_symbol_pairs):
            s1, s2 = s1_vec[i], s2_vec[i]

            for rx in range(n_rx_antennas):
                h1, h2 = h[rx, 0, i], h[rx, 1, i]  # channels from tx1 and tx2 to rx

                # Time slot 1: [s1, s2] transmitted from [tx1, tx2]
                r1[rx, i] = h1 * s1 + h2 * s2 + noise[rx, 0, i]

                # Time slot 2: [-s2*, s1*] transmitted from [tx1, tx2]
                # For real BPSK: s* = s
                r2[rx, i] = h1 * (-s2) + h2 * s1 + noise[rx, 1, i]

        # Alamouti decoding
        s1_hat = np.zeros(n_symbol_pairs, dtype=complex)
        s2_hat = np.zeros(n_symbol_pairs, dtype=complex)

        for i in range(n_symbol_pairs):
            s1_sum = 0
            s2_sum = 0

            for rx in range(n_rx_antennas):
                h1, h2 = h[rx, 0, i], h[rx, 1, i]

                # Alamouti combining
                s1_sum += np.conj(h1) * r1[rx, i] + h2 * np.conj(r2[rx, i])
                s2_sum += np.conj(h2) * r1[rx, i] - h1 * np.conj(r2[rx, i])

            s1_hat[i] = s1_sum
            s2_hat[i] = s2_sum

        # Decision: take real part and apply a threshold
        decoded_symbols = np.zeros(n_bits)
        decoded_symbols[0::2] = np.real(s1_hat)  # s1 estimates
        decoded_symbols[1::2] = np.real(s2_hat)  # s2 estimates

        decoded_bits = np.where(decoded_symbols >= 0, 1, 0)

        return random_bits, decoded_bits

    @staticmethod
    def simulate(
            n_bits: int,
            variance: float,
            apply_fading: bool = True,
            n_antennas: int = 1,
            use_alamouti: bool = False
    ) -> tuple:

        if use_alamouti:
            return BPSKSimulation.simulate_alamouti(n_bits, variance, n_antennas)

        random_bits = BPSKSimulation.generate_random_bits(n_bits)
        coded_bits = BPSKSimulation.apply_coding(random_bits)

        if apply_fading:
            if n_antennas == 1:
                # Single antenna with fading
                fading = BPSKSimulation.generate_rayleigh_fading(n_bits)
                received_signal = fading * coded_bits
                noise = BPSKSimulation.generate_gaussian_noise(n_bits, variance)
                received_signal += noise
                received_signal = BPSKSimulation.thresholding(received_signal, threshold=0.0)
            else:
                mrc_signal = np.zeros(n_bits, dtype=float)
                for _ in range(n_antennas):
                    h = BPSKSimulation.generate_rayleigh_fading(n_bits)
                    noise = BPSKSimulation.generate_gaussian_noise(n_bits, variance)
                    r = h * coded_bits + noise
                    mrc_signal += h * r  # MRC combining
                received_signal = BPSKSimulation.thresholding(mrc_signal, threshold=0.0)
        else:
            # AWGN only
            received_signal = BPSKSimulation.apply_awgn_channel(coded_bits, variance)
            received_signal = BPSKSimulation.thresholding(received_signal, threshold=0.0)

        decoded_bits = BPSKSimulation.apply_decoding(received_signal)
        return random_bits, decoded_bits

    @staticmethod
    def run_monte_carlo_simulation(
            n_bits: int,
            snr_db_values: np.ndarray,
            apply_fading: bool = True,
            n_antennas: int = 1,
            use_alamouti: bool = False,
            n_trials: int = 25
    ) -> np.ndarray:

        ber_results = []
        snr_linear = 10 ** (snr_db_values / 10)

        for i, snr_val in enumerate(snr_linear):
            errors = []
            for trial in range(n_trials):
                if use_alamouti:
                    variance = 2 / snr_val
                else:
                    variance = 1 / snr_val
                random_bits, decoded_bits = BPSKSimulation.simulate(
                    n_bits, variance, apply_fading, n_antennas, use_alamouti
                )
                errors.append(BPSKSimulation.calculate_error(random_bits, decoded_bits))
            ber = np.mean(errors)
            ber_results.append(ber)
            print(f"SNR: {snr_db_values[i]:2.0f} dB, BER: {ber:.6f}")
        return np.array(ber_results)