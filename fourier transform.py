import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import os

def create_custom_function(x):
    return np.sin(x) + np.cos(2*x) + 0.5*np.sin(5*x)

class FourierAnimator:
    """Animate Fourier transform decomposition for any function."""
    
    def __init__(self, x_vals, func_vals, label="Function"):
        """
        Args:
            x_vals: array of x values (time or space)
            func_vals: array of function values
            label: name of the function for display
        """
        self.x = x_vals
        self.f = func_vals
        self.label = label
        self.N = len(x_vals)
        
        self.fft = np.fft.fft(func_vals)
        self.freqs = np.fft.fftfreq(self.N, self.x[1] - self.x[0])
        self.magnitude = np.abs(self.fft)
        self.phase = np.angle(self.fft)
        
        self.pos_idx = self.freqs > 0
        self.freqs_pos = self.freqs[self.pos_idx]
        self.mag_pos = self.magnitude[self.pos_idx]
        self.phase_pos = self.phase[self.pos_idx]
        
        sort_idx = np.argsort(self.freqs_pos)
        self.freqs_pos = self.freqs_pos[sort_idx]
        self.mag_pos = self.mag_pos[sort_idx]
        self.phase_pos = self.phase_pos[sort_idx]
        
        self.num_freqs = len(self.freqs_pos)
    
    def reconstruct_partial(self, num_freqs_include):
        """Reconstruct signal using only first num_freqs_include frequency components."""
        fft_partial = np.zeros_like(self.fft, dtype=complex)
        
        fft_partial[0] = self.fft[0]
        
        for i in range(num_freqs_include):
            freq_target = self.freqs_pos[i]
            pos_match = np.argmin(np.abs(self.freqs - freq_target))
            fft_partial[pos_match] = self.fft[pos_match]
            neg_match = np.argmin(np.abs(self.freqs + freq_target))
            fft_partial[neg_match] = self.fft[neg_match]
        
        reconstructed = np.fft.ifft(fft_partial).real
        return reconstructed
    
    def get_frame_data(self, frame_idx):
        """Get data for a specific animation frame."""
        num_freqs = int((frame_idx / 100.0) * self.num_freqs)  
        num_freqs = max(1, min(num_freqs, self.num_freqs))
        reconstructed = self.reconstruct_partial(num_freqs)
        
        return {
            'num_freqs': num_freqs,
            'reconstructed': reconstructed,
            'error': np.linalg.norm(self.f - reconstructed)
        }

def plot_fourier_frame(fig, animator, frame_idx, total_frames, title_prefix=""):
    """Plot the animation frame showing original, spectrum, and reconstruction."""
    fig.clear()
    
    frame_data = animator.get_frame_data(frame_idx)
    num_freqs = frame_data['num_freqs']
    reconstructed = frame_data['reconstructed']
    error = frame_data['error']
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax1.plot(animator.x, animator.f, 'b-', linewidth=2.5, label='Original Signal')
    ax1.fill_between(animator.x, animator.f, alpha=0.3, color='blue')
    ax1.set_xlabel('Time/Space', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
    ax1.set_title('Original Function', fontsize=11, fontweight='bold', color='darkblue')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    ax2.bar(range(num_freqs), animator.mag_pos[:num_freqs], color='cyan', 
            edgecolor='darkblue', linewidth=1.2, alpha=0.8)
    if num_freqs < animator.num_freqs:
        ax2.bar(range(num_freqs, animator.num_freqs), 
                animator.mag_pos[num_freqs:], color='lightgrey', 
                edgecolor='grey', linewidth=0.8, alpha=0.4)
    ax2.set_xlabel('Frequency Component', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Magnitude', fontsize=10, fontweight='bold')
    ax2.set_title(f'Magnitude Spectrum (Using {num_freqs}/{animator.num_freqs})', 
                  fontsize=11, fontweight='bold', color='darkgreen')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(0, animator.num_freqs)
    
    ax3.scatter(range(num_freqs), animator.phase_pos[:num_freqs], 
                c='magenta', s=60, edgecolor='darkmagenta', linewidth=1.2, alpha=0.8, label='Active')
    if num_freqs < animator.num_freqs:
        ax3.scatter(range(num_freqs, animator.num_freqs), 
                    animator.phase_pos[num_freqs:], 
                    c='lightgrey', s=40, edgecolor='grey', linewidth=0.8, alpha=0.4, label='Inactive')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(np.pi, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(-np.pi, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Frequency Component', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Phase (radians)', fontsize=10, fontweight='bold')
    ax3.set_title(f'Phase Spectrum', fontsize=11, fontweight='bold', color='darkred')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(0, animator.num_freqs)
    ax3.set_ylim(-np.pi - 0.5, np.pi + 0.5)
    
    ax4.plot(animator.x, animator.f, 'b-', linewidth=2.5, label='Original', alpha=0.7)
    ax4.plot(animator.x, reconstructed, 'r--', linewidth=2.5, label='Reconstruction')
    ax4.fill_between(animator.x, animator.f, reconstructed, alpha=0.25, color='orange')
    ax4.set_xlabel('Time/Space', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
    ax4.set_title(f'Reconstruction Error: {error:.4e}', fontsize=11, fontweight='bold', color='darkred')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=9)
    
    fig.suptitle(f'{title_prefix} Fourier Transform Animation\nFrame {frame_idx+1}/{total_frames} | '
                 f'Using {num_freqs} of {animator.num_freqs} frequency components',
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def create_square_wave(x, period=2.0, amplitude=1.0):
    """Square wave function."""
    return amplitude * np.sign(np.sin(2 * np.pi * x / period))

def create_sawtooth_wave(x, period=2.0, amplitude=1.0):
    """Sawtooth wave function."""
    return amplitude * 2 * (x / period - np.floor(x / period + 0.5))

def create_triangle_wave(x, period=2.0, amplitude=1.0):
    """Triangle wave function."""
    x_norm = (x % period) / period
    return amplitude * (4 * np.where(x_norm < 0.5, x_norm, 1 - x_norm) - 1)

def create_composite_signal(x):
    """Composite signal: sum of sine waves at different frequencies."""
    return (1.0 * np.sin(2 * np.pi * x) + 
            0.5 * np.sin(2 * np.pi * 3 * x) + 
            0.3 * np.sin(2 * np.pi * 5 * x) +
            0.2 * np.cos(2 * np.pi * 2 * x) +
            0.1 * np.random.randn(len(x)))  # Add noise

def create_pulse_train(x, pulse_width=0.2, period=1.0, amplitude=1.0):
    """Pulse train (rectangle pulses)."""
    x_norm = (x % period) / period
    return amplitude * np.where(x_norm < pulse_width, 1.0, 0.0)

def create_chirp_signal(x, f0=1, f1=5, amplitude=1.0):
    """Chirp signal (frequency sweep)."""
    return amplitude * np.sin(2 * np.pi * (f0 * x + (f1 - f0) * x**2 / (2 * np.max(x))))

def create_modulated_signal(x, carrier_freq=10, modulation_freq=2, amplitude=1.0):
    """Amplitude modulated signal."""
    modulation = 1 + 0.7 * np.sin(2 * np.pi * modulation_freq * x)
    return amplitude * modulation * np.sin(2 * np.pi * carrier_freq * x)

if __name__ == "__main__":
    print("=" * 80)
    print("FOURIER TRANSFORM ANIMATOR - Dynamic Frequency Buildup Visualization")
    print("=" * 80)
    
    current_dir = os.getcwd()
    print(f"\nSaving to: {current_dir}\n")
    
    x = np.linspace(0, 4, 512)  # Domain: 0 to 4
    
    print("Available functions:")
    print("  1. Square Wave")
    print("  2. Sawtooth Wave")
    print("  3. Triangle Wave")
    print("  4. Composite Signal (sum of sine waves + noise)")
    print("  5. Pulse Train")
    print("  6. Chirp Signal (frequency sweep)")
    print("  7. Amplitude Modulated Signal")
    
    choice = 4 
    
    if choice == 1:
        f = create_square_wave(x)
        func_name = "Square Wave"
    elif choice == 2:
        f = create_sawtooth_wave(x)
        func_name = "Sawtooth Wave"
    elif choice == 3:
        f = create_triangle_wave(x)
        func_name = "Triangle Wave"
    elif choice == 4:
        np.random.seed(42)
        f = create_composite_signal(x)
        func_name = "Composite Signal (3-sine + noise)"
    elif choice == 5:
        f = create_pulse_train(x)
        func_name = "Pulse Train"
    elif choice == 6:
        f = create_chirp_signal(x)
        func_name = "Chirp Signal"
    else:
        f = create_modulated_signal(x)
        func_name = "Amplitude Modulated Signal"
    
    animator = FourierAnimator(x, f, label=func_name)
    
    print(f"\n✓ Function: {func_name}")
    print(f"✓ Domain samples: {len(x)}")
    print(f"✓ Frequency components: {animator.num_freqs}")
    
    print("\nCreating final state visualization...")
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    plot_fourier_frame(fig, animator, animator.num_freqs - 1, animator.num_freqs, 
                       title_prefix=func_name)
    
    png_path = os.path.join(current_dir, f'fourier_{func_name.replace(" ", "_").lower()}_final.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: fourier_*_final.png ({os.path.getsize(png_path)/(1024*1024):.2f} MB)")
    
    print(f"\nCreating animation (100 frames)...")
    print("(This will take ~1-2 minutes...)\n")
    
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    total_frames = 100
    
    def update(frame_idx):
        plot_fourier_frame(fig, animator, frame_idx, total_frames, 
                          title_prefix=func_name)
        return fig.get_axes()
    
    anim = FuncAnimation(fig, update, frames=range(total_frames),
                        interval=50, repeat=True, repeat_delay=1000)
    
    gif_path = os.path.join(current_dir, f'fourier_{func_name.replace(" ", "_").lower()}_animation.gif')
    
    try:
        print("Encoding animation (please wait)...")
        writer = PillowWriter(fps=20)
        anim.save(gif_path, writer=writer, dpi=100)
        plt.close()
        size_mb = os.path.getsize(gif_path) / (1024*1024)
        print(f"✓ Saved: fourier_*_animation.gif ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"✗ Error saving GIF: {e}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("OUTPUT FILES:")
    print("=" * 80)
    files = [f for f in os.listdir(current_dir) if f.startswith('fourier') and f.endswith(('.png', '.gif'))]
    for f in sorted(files):
        size = os.path.getsize(os.path.join(current_dir, f)) / (1024*1024)
        print(f"  ✓ {f:50s} ({size:6.2f} MB)")
    
    print("\n" + "=" * 80)
    print("FOURIER TRANSFORM ANIMATION COMPLETE!")
    print("=" * 80)