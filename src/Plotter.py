from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import os


class Plotter:
    def __init__(self):
        # Professional color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#7209B7',
            'dark': '#2D3748',
            'light': '#F7FAFC'
        }

        # Professional markers
        self.markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

        # Setup matplotlib parameters for publication quality
        self.setup_matplotlib()

    def setup_matplotlib(self):
        """Configure matplotlib for professional plots with better readability"""
        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
            'axes.linewidth': 1.2,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 6,
            'xtick.minor.size': 4,
            'ytick.major.size': 6,
            'ytick.minor.size': 4,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True,
            'legend.numpoints': 1,
            'legend.scatterpoints': 1,
            'legend.fontsize': 13,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.8,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.unicode_minus': False
        })

    def _ensure_image_directory(self):
        """Create Image directory if it doesn't exist"""
        if not os.path.exists('Image'):
            os.makedirs('Image')

    def plot_single_channel(self, snr_db, errors_sim, errors_theory, channel_type,
                            figsize=(12, 8), save_path=None):
        """Plot BER curve for a single channel type with professional styling"""

        fig, ax = plt.subplots(figsize=figsize)

        # Plot simulation results
        ax.semilogy(snr_db, errors_sim, 'o-',
                    color=self.colors['primary'],
                    markersize=8, linewidth=3,
                    markerfacecolor='white',
                    markeredgewidth=2.5,
                    label=f'{channel_type} (Simulation)',
                    alpha=0.9)

        # FIX: Conditionally plot the theoretical curve only if data is provided.
        if errors_theory is not None:
            ax.semilogy(snr_db, errors_theory, '--',
                        color=self.colors['secondary'],
                        linewidth=3.5,
                        label=f'{channel_type} (Theoretical)',
                        alpha=0.8)

        # Styling with larger fonts
        ax.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Bit Error Rate (BER)', fontsize=16, fontweight='bold')
        ax.set_title(f'BPSK Performance over {channel_type} Channel',
                     fontsize=18, fontweight='bold', pad=25)

        # Grid and limits
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim([1e-6, 1])
        ax.set_xlim([min(snr_db), max(snr_db)])

        # Legend will now automatically adjust based on what was plotted
        legend = ax.legend(loc='upper right', fontsize=14,
                           framealpha=0.9, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')

        # Minor ticks
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        plt.tight_layout()

        # Auto-save to Image directory
        self._ensure_image_directory()
        if save_path is None:
            save_path = f'Image/{channel_type}_single_channel.png'
        elif not save_path.startswith('Image/'):
            save_path = f'Image/{save_path}'

        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

        plt.show()

    def plot_comparison_advanced(self, snr_db_list, errors_list, errors_theoretical_list,
                                 channel_types, figsize=(16, 10), save_path=None):
        """Advanced comparison plot with professional styling"""

        fig, ax = plt.subplots(figsize=figsize)

        colors_list = [self.colors['primary'], self.colors['secondary'],
                       self.colors['accent'], self.colors['success']]

        for i, channel_type in enumerate(channel_types):
            color = colors_list[i % len(colors_list)]
            marker = self.markers[i % len(self.markers)]

            # Simulation results
            ax.semilogy(snr_db_list[i], errors_list[i],
                        marker=marker, linestyle='-', color=color,
                        markersize=8, linewidth=3,
                        markerfacecolor='white', markeredgewidth=2.5,
                        label=f'{channel_type} (Simulation)',
                        alpha=0.9, markevery=2)

            # FIX: Conditionally plot theoretical results for each channel type.
            # This handles cases where the entire list or individual items are None.
            if errors_theoretical_list is not None and errors_theoretical_list[i] is not None:
                ax.semilogy(snr_db_list[i], errors_theoretical_list[i],
                            linestyle='--', color=color, linewidth=3.5,
                            label=f'{channel_type} (Theory)',
                            alpha=0.7)

        # Professional styling
        ax.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Bit Error Rate (BER)', fontsize=18, fontweight='bold')
        ax.set_title('BPSK Performance Comparison: Channel Types',
                     fontsize=20, fontweight='bold', pad=30)

        # Grid and axis formatting
        ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_ylim([1e-7, 1])

        all_snr = np.concatenate(snr_db_list)
        ax.set_xlim([min(all_snr), max(all_snr)])

        # Professional legend
        legend = ax.legend(loc='upper right', fontsize=14,
                           ncol=2, framealpha=0.95,
                           fancybox=True, shadow=True,
                           columnspacing=1.5, handletextpad=0.5)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1)

        # Minor ticks
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(which='major', length=6, width=1.2)
        ax.tick_params(which='minor', length=3, width=0.8)

        # Add performance annotations
        self._add_performance_annotations(ax, snr_db_list, errors_list, channel_types)

        plt.tight_layout()

        # Auto-save to Image directory
        self._ensure_image_directory()
        if save_path is None:
            save_path = 'Image/Simple_comparison.png'
        elif not save_path.startswith('Image/'):
            save_path = f'Image/{save_path}'

        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

        plt.show()

    def _add_performance_annotations(self, ax, snr_db_list, errors_list, channel_types):
        """Add performance annotations to the plot based on simulation results."""
        target_ber = 1e-3

        for i, (snr_db, errors, channel_type) in enumerate(zip(snr_db_list, errors_list, channel_types)):
            errors_array = np.array(errors)
            # Find index where simulation error is closest to the target BER
            idx = np.argmin(np.abs(errors_array - target_ber))

            # Add annotation if the point is reasonably close to the target
            if errors_array[idx] <= target_ber * 2 and errors_array[idx] > 0:
                snr_at_target = snr_db[idx]
                ax.annotate(f'{channel_type}\n≈{snr_at_target:.1f} dB @ 10⁻³',
                            xy=(snr_at_target, target_ber),
                            xytext=(snr_at_target + 3, target_ber * 10),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7, lw=1.5),
                            fontsize=12, ha='left', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.4',
                                      facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.2))

    def create_subplot_comparison(self, snr_db_list, errors_list, errors_theoretical_list,
                                  channel_types, figsize=(18, 8), save_path=None):
        """Create side-by-side subplot comparison"""
        if not channel_types:
            print("No data to plot.")
            return

        fig, axes = plt.subplots(1, len(channel_types), figsize=figsize, sharey=True)

        if len(channel_types) == 1:
            axes = [axes]

        colors_list = [self.colors['primary'], self.colors['secondary']]

        for i, (ax, channel_type) in enumerate(zip(axes, channel_types)):
            color = colors_list[i % len(colors_list)]

            # Plot simulation curves
            ax.semilogy(snr_db_list[i], errors_list[i], 'o-',
                        color=color, markersize=7, linewidth=3,
                        markerfacecolor='white', markeredgewidth=2.5,
                        label='Simulation', alpha=0.9)

            # FIX: Conditionally plot theoretical curves for each subplot.
            if errors_theoretical_list is not None and errors_theoretical_list[i] is not None:
                ax.semilogy(snr_db_list[i], errors_theoretical_list[i], '--',
                            color=color, linewidth=3.5, alpha=0.7,
                            label='Theoretical')

            # Individual subplot styling
            ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
            if i == 0:
                ax.set_ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')

            ax.set_title(f'{channel_type} Channel', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            ax.set_ylim([1e-6, 1])

        plt.suptitle('BPSK Performance: Individual Channel Analysis',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to prevent suptitle overlap

        # Auto-save to Image directory
        self._ensure_image_directory()
        if save_path is None:
            save_path = 'Image/BPSK_subplot_comparison.png'
        elif not save_path.startswith('Image/'):
            save_path = f'Image/{save_path}'

        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

        plt.show()