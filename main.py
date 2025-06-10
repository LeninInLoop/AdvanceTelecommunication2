from src.BPSKSimulator import BPSKSimulation, np
from src.TheoreticalCalculations import TheoreticalCalculations
from src.Plotter import Plotter


def main():
    plotter = Plotter()

    snr_db_values = np.linspace(0, 30, 31)
    snr_linear_values = 10 ** (snr_db_values / 10)

    print("Simulating BPSK Coherent With Rayleigh Fading (Single Antenna)...")
    errors_fading = BPSKSimulation.run_monte_carlo_simulation(
        n_bits=10 ** 6,
        snr_db_values=snr_db_values,
        apply_fading=True,
    )

    errors_theoretical_fading = TheoreticalCalculations.bpsk_coherent_theoretical_ber_rayleigh_fading(snr_linear_values)

    plotter.plot_single_channel(
        snr_db_values, errors_fading, errors_theoretical_fading, 'Simple Rayleigh Fading'
    )

    print("Simulating BPSK Coherent With AWGN Channel (Single Antenna)...")
    errors_awgn = BPSKSimulation.run_monte_carlo_simulation(
        n_bits=10 ** 6,
        snr_db_values=snr_db_values,
        apply_fading=False,
    )
    errors_theoretical_awgn = TheoreticalCalculations.bpsk_coherent_theoretical_ber_awgn(snr_linear_values)

    plotter.plot_single_channel(
        snr_db_values, errors_awgn, errors_theoretical_awgn, 'AWGN'
    )

    plotter.plot_comparison_advanced(
        snr_db_list=[snr_db_values, snr_db_values],
        errors_list=[errors_awgn, errors_fading],
        errors_theoretical_list=[errors_theoretical_awgn, errors_theoretical_fading],
        channel_types=['AWGN', 'Simple Rayleigh Fading'],
        save_path="simple_rayleigh_fading_and_simple_awgn_comparison.png"
    )

    # MRC Simulation and Plotting
    print("Simulating BPSK Coherent With MRC ...")
    antenna_configs = [2, 4, 8]

    mrc_errors_list = []
    mrc_theoretical_list = []
    mrc_channel_types = []

    for n_antennas in antenna_configs:
        print(f"Simulating MRC with {n_antennas} antenna(s) ...")

        # Simulation
        errors_mrc = BPSKSimulation.run_monte_carlo_simulation(
            n_bits=10 ** 6,
            snr_db_values=snr_db_values,
            apply_fading=True,
            n_antennas=n_antennas
        )

        # Theoretical
        errors_theoretical_mrc = TheoreticalCalculations.bpsk_coherent_theoretical_ber_mrc(
            snr_linear_values,
            n_antennas
        )

        mrc_errors_list.append(errors_mrc)
        mrc_theoretical_list.append(errors_theoretical_mrc)
        mrc_channel_types.append(f'MRC {n_antennas} Antenna{"s" if n_antennas > 1 else ""}')

        # Individual plot for each MRC configuration
        plotter.plot_single_channel(
            snr_db_values, errors_mrc, errors_theoretical_mrc,
            f'MRC {n_antennas} Antenna{"s" if n_antennas > 1 else ""}'
        )

    # MRC Comparison plot
    snr_db_list_mrc = [snr_db_values] * len(antenna_configs)

    plotter.plot_comparison_advanced(
        snr_db_list=snr_db_list_mrc,
        errors_list=mrc_errors_list,
        errors_theoretical_list=mrc_theoretical_list,
        channel_types=mrc_channel_types,
        save_path="MRC_comparison.png"
    )

    # MRC and Simple Comparison
    combined_snr_db_list = snr_db_list_mrc + [snr_db_values] + [snr_db_values]
    combined_errors_list = mrc_errors_list + [errors_awgn] + [errors_fading]
    combined_theoretical_list = mrc_theoretical_list + [errors_theoretical_awgn] + [errors_theoretical_fading]
    combined_channel_types = mrc_channel_types + ['AWGN'] + ["Simple Rayleigh Fading"]

    plotter.plot_comparison_advanced(
        snr_db_list=combined_snr_db_list,
        errors_list=combined_errors_list,
        errors_theoretical_list=combined_theoretical_list,
        channel_types=combined_channel_types,
        save_path="MRC_&_AWGN_comparison.png"
    )

    # alamouti Simulation
    print("Simulating BPSK Coherent With alamouti ...")
    antenna_configs = [1, 2, 4, 8]

    alamouti_errors_list = []
    alamouti_channel_types = []

    for n_antennas in antenna_configs:
        print(f"Simulating alamouti with {n_antennas} receiver antenna(s) ...")

        # Simulation
        errors_alamouti = BPSKSimulation.run_monte_carlo_simulation(
            n_bits=10 ** 6,
            snr_db_values=snr_db_values,
            apply_fading=True,
            use_alamouti=True,
            n_antennas=n_antennas
        )

        alamouti_errors_list.append(errors_alamouti)

        alamouti_channel_types.append(f'Alamouti, 2-TX {n_antennas}-Rx')

        # Individual plot for each MRC configuration
        plotter.plot_single_channel(
            snr_db_values, errors_alamouti, None,
            f'Alamouti, 2-TX {n_antennas}-Rx'
        )

    # MRC Comparison plot
    snr_db_list_alamouti = [snr_db_values] * len(antenna_configs)

    plotter.plot_comparison_advanced(
        snr_db_list=snr_db_list_alamouti,
        errors_list=alamouti_errors_list,
        errors_theoretical_list=None,
        channel_types=alamouti_channel_types,
        save_path="alamouti_comparison.png"
    )

    # Full Comparison
    combined_snr_db_list = snr_db_list_mrc + [snr_db_values] + [snr_db_values] + snr_db_list_alamouti
    combined_errors_list = mrc_errors_list + [errors_awgn] + [errors_fading] + alamouti_errors_list
    combined_theoretical_list = None
    combined_channel_types = mrc_channel_types + ['AWGN'] + ["Simple Rayleigh Fading"] + alamouti_channel_types

    plotter.plot_comparison_advanced(
        snr_db_list=combined_snr_db_list,
        errors_list=combined_errors_list,
        errors_theoretical_list=combined_theoretical_list,
        channel_types=combined_channel_types,
        save_path="Alamouti_MRC_&_AWGN_comparison.png"
    )


if __name__ == '__main__':
    main()
