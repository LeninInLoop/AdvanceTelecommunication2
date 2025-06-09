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
        # Complex Rayleigh fading (real and imaginary parts are independent Gaussian)
        real = np.random.randn(*shape) * np.sqrt(0.5)
        imag = np.random.randn(*shape) * np.sqrt(0.5)
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
    def simulate_alamouti(n_bits: int, variance: float, n_antennas: int) -> tuple:

        n_bits = n_bits if n_bits % 2 == 0 else n_bits - 1
        n_symbols = n_bits // 2

        random_bits = BPSKSimulation.generate_random_bits(n_bits)
        symbols = BPSKSimulation.apply_coding(random_bits).reshape((n_symbols, 2))

        # Generate complex Rayleigh fading channels (n_antennas x 2)
        H = BPSKSimulation.generate_complex_rayleigh_fading((n_antennas, 2, n_symbols))

        # Generate complex AWGN noise
        noise = BPSKSimulation.generate_complex_gaussian_noise((n_symbols, n_antennas, 2), variance)

        # Transmit Alamouti encoded symbols
        Y = np.zeros((n_symbols, n_antennas, 2), dtype=complex)
        for i in range(n_symbols):
            s1, s2 = symbols[i]
            for j in range(n_antennas):
                h1, h2 = H[j, :, i]
                # Time slot 1: r1 = h1*s1 + h2*s2
                Y[i, j, 0] = h1 * s1 + h2 * s2 + noise[i, j, 0]
                # Time slot 2: r2 = h1*(-s2*) + h2*(s1*) = -h1*s2 + h2*s1 (since BPSK is real)
                Y[i, j, 1] = -h1 * s2 + h2 * s1 + noise[i, j, 1]

        # Alamouti decoding
        s_hat = np.zeros((n_symbols, 2), dtype=complex)
        for i in range(n_symbols):
            s1_hat = 0
            s2_hat = 0
            for j in range(n_antennas):
                h1, h2 = H[j, :, i]
                r1 = Y[i, j, 0]
                r2 = Y[i, j, 1]

                # Alamouti combining
                s1_hat += np.conj(h1) * r1 + h2 * np.conj(r2)
                s2_hat += np.conj(h2) * r1 - h1 * np.conj(r2)

            s_hat[i, 0] = s1_hat
            s_hat[i, 1] = s2_hat

        # Decision based on real part
        decoded_symbols = s_hat.reshape(-1)
        decoded_bits = BPSKSimulation.thresholding_complex(decoded_symbols)

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
                coded_bits = BPSKSimulation.apply_fading(coded_bits)
            else:
                mrc_combined_signal = np.zeros_like(coded_bits, dtype=float)
                for _ in range(n_antennas):
                    h = BPSKSimulation.generate_rayleigh_fading(n_bits)
                    r_i = h * coded_bits + BPSKSimulation.generate_gaussian_noise(n_bits, variance)
                    mrc_combined_signal += h * r_i
                coded_bits = mrc_combined_signal

        if not apply_fading or n_antennas == 1:
            noisy_bits = BPSKSimulation.apply_awgn_channel(coded_bits, variance)
        else:
            noisy_bits = coded_bits

        extracted_bits = BPSKSimulation.thresholding(noisy_bits, threshold=0)
        decoded_bits = BPSKSimulation.apply_decoding(extracted_bits)
        return random_bits, decoded_bits

    @staticmethod
    def run_monte_carlo_simulation(
            n_bits: int,
            snr_db_values: np.ndarray,
            apply_fading: bool = True,
            n_antennas: int = 1,
            use_alamouti: bool = False
    ) -> np.ndarray:

        errors = []
        snr_linear = 10 ** (snr_db_values / 10)

        for snr_val in snr_linear:
            pre_errors = []
            for _ in range(20):
                if use_alamouti:
                    variance = 1 / snr_val
                else:
                    variance = 2 / snr_val

                random_bits, decoded_bits = BPSKSimulation.simulate(
                    n_bits, variance, apply_fading, n_antennas, use_alamouti
                )
                pre_errors.append(BPSKSimulation.calculate_error(random_bits, decoded_bits))
            errors.append(np.mean(pre_errors))
            print(errors)
        return errors