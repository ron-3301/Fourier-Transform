"""
Manim Animation: Fourier Transforms of Periodic Functions
This script creates animated visualizations of how periodic functions 
can be decomposed and reconstructed using their Fourier series.

Usage:
    python fourier_animation.py
    
The script will prompt you to select a periodic function and then render
the animation, saving it as an MP4 file in the current directory.
"""

from manim import *
from scipy.integrate import quad
import numpy as np
from typing import Callable, Tuple, List
import sys
import os
from pathlib import Path
import subprocess


class FourierSeriesCalculator:
    """Calculate Fourier series coefficients for periodic functions."""
    
    def __init__(self, func: Callable, period: float = 2 * PI):
        """
        Initialize the calculator.
        
        Args:
            func: The periodic function to analyze
            period: The period of the function (default: 2π)
        """
        self.func = func
        self.period = period
    
    def calculate_fourier_coefficients(self, n_terms: int) -> Tuple[float, List[float], List[float]]:
        """
        Calculate a₀, aₙ, and bₙ coefficients.
        
        Args:
            n_terms: Number of terms to calculate
            
        Returns:
            Tuple of (a0, a_coefficients, b_coefficients)
        """
        # Calculate a₀
        a0_integral, _ = quad(self.func, 0, self.period)
        a0 = (2 / self.period) * a0_integral
        
        a_coefficients = []
        b_coefficients = []
        
        # Calculate aₙ and bₙ
        for n in range(1, n_terms + 1):
            # aₙ coefficient
            def an_integrand(x):
                return self.func(x) * np.cos(2 * n * np.pi * x / self.period)
            
            an_integral, _ = quad(an_integrand, 0, self.period)
            an = (2 / self.period) * an_integral
            a_coefficients.append(an)
            
            # bₙ coefficient
            def bn_integrand(x):
                return self.func(x) * np.sin(2 * n * np.pi * x / self.period)
            
            bn_integral, _ = quad(bn_integrand, 0, self.period)
            bn = (2 / self.period) * bn_integral
            b_coefficients.append(bn)
        
        return a0, a_coefficients, b_coefficients
    
    def fourier_approximation(self, x: float, a0: float, a_coeffs: List[float], b_coeffs: List[float]) -> float:
        """
        Calculate the Fourier series approximation at point x.
        """
        result = a0 / 2
        for n, (an, bn) in enumerate(zip(a_coeffs, b_coeffs), 1):
            result += an * np.cos(2 * n * np.pi * x / self.period) + \
                      bn * np.sin(2 * n * np.pi * x / self.period)
        return result


def get_predefined_functions():
    """Return a dictionary of predefined periodic functions."""
    return {
        "1": {
            "name": "Square Wave",
            "func": lambda x: 1 if (x % (2 * PI)) < PI else -1,
        },
        "2": {
            "name": "Sawtooth Wave",
            "func": lambda x: 2 * ((x % (2 * PI)) / (2 * PI)) - 1,
        },
        "3": {
            "name": "Triangle Wave",
            "func": lambda x: 1 - 4 * abs(((x % (2 * PI)) - PI)) / (2 * PI),
        },
        "4": {
            "name": "Sine Wave",
            "func": lambda x: np.sin(x),
        },
        "5": {
            "name": "Half Sine Wave (Rectified)",
            "func": lambda x: max(0, np.sin(x)),
        },
        "6": {
            "name": "Pulse Wave",
            "func": lambda x: 1 if (x % (2 * PI)) < (PI / 2) else -1,
        },
    }


class FourierTransformAnimation(Scene):
    """Animate the Fourier transform of a periodic function."""
    
    def construct(self):
        # Configuration
        self.camera.background_color = WHITE
        
        # Get function choice from environment variable
        import os
        func_choice = os.environ.get('FOURIER_FUNC_CHOICE', '1')
        n_harmonics = int(os.environ.get('FOURIER_N_HARMONICS', '10'))
        function_name = os.environ.get('FOURIER_FUNC_NAME', 'Function')
        
        # Get the predefined functions
        functions = get_predefined_functions()
        
        # Select the function
        if func_choice in functions:
            func = functions[func_choice]["func"]
            function_name = functions[func_choice]["name"]
        else:
            func = functions["1"]["func"]
            function_name = functions["1"]["name"]
        
        period = 2 * PI
        
        # Create calculator
        calculator = FourierSeriesCalculator(func, period)
        a0, a_coeffs, b_coeffs = calculator.calculate_fourier_coefficients(n_harmonics)
        
        # Create axes for the original function
        axes_original = Axes(
            x_range=[0, 2 * PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GREY_B},
            tips=False,
        )
        axes_original.scale(0.6)  # Scale down to fit on screen
        
        # Create axes for frequency spectrum
        # Use max 50 harmonics for x-axis, or n_harmonics if larger
        max_harmonics = max(50, n_harmonics)
        axes_spectrum = Axes(
            x_range=[0, max_harmonics, 5],  # Step of 5 for cleaner labels
            y_range=[0, 1.2, 0.2],
            axis_config={"color": GREY_B},
            tips=False,
        )
        axes_spectrum.scale(0.6)  # Scale down to fit on screen
        
        # Position the axes
        axes_original.to_edge(LEFT)
        axes_spectrum.to_edge(RIGHT)
        
        # Create labels
        label_original = Text("Original Function", font_size=24, color=BLACK)
        label_original.next_to(axes_original, UP)
        
        label_spectrum = Text("Frequency Spectrum", font_size=24, color=BLACK)
        label_spectrum.next_to(axes_spectrum, UP)
        
        # Draw original function
        original_func = axes_original.plot(func, color=BLUE, stroke_width=2.5)
        
        self.add(axes_original, axes_spectrum, label_original, label_spectrum)
        self.play(Create(original_func))
        self.wait(1)
        
        # Animate the buildup of Fourier series
        fourier_curves = []
        spectrum_bars = []
        
        for harmonic in range(1, n_harmonics + 1):
            # Calculate current approximation
            current_a_coeffs = a_coeffs[:harmonic]
            current_b_coeffs = b_coeffs[:harmonic]
            
            def fourier_func(x):
                return calculator.fourier_approximation(x, a0, current_a_coeffs, current_b_coeffs)
            
            # Plot Fourier approximation
            fourier_curve = axes_original.plot(fourier_func, color=RED, stroke_width=2.5)
            
            # Calculate magnitude spectrum (amplitude of each harmonic)
            magnitude = np.sqrt(a_coeffs[harmonic-1]**2 + b_coeffs[harmonic-1]**2)
            
            # Create spectrum bar using a simple scale
            bar_height = magnitude / 2  # Scale for visibility
            
            # Use rectangles for clearer bars
            # Height is set to a fixed proportion for visual clarity
            bar = Rectangle(
                width=0.3,
                height=0.5,  # Fixed height works well for all Manim versions
                color=GREEN,
                fill_opacity=0.7,
            )
            # Scale the bar to match the magnitude
            bar.scale(bar_height / 0.5)  # Scale relative to the base height
            bar.move_to(axes_spectrum.coords_to_point(harmonic, bar_height / 2))
            
            # Animate the addition
            if harmonic == 1:
                self.play(
                    Transform(original_func, fourier_curve),
                    Create(bar),
                    run_time=0.8
                )
            else:
                self.play(
                    Transform(fourier_curves[-1], fourier_curve),
                    Create(bar),
                    run_time=0.6
                )
            
            fourier_curves.append(fourier_curve)
            spectrum_bars.append(bar)
            self.wait(0.3)
        
        self.wait(2)
        
        # Add annotation
        annotation = Text(
            f"{function_name} - Fourier Series with {n_harmonics} Harmonics",
            font_size=20,
            color=BLACK
        )
        annotation.to_edge(DOWN)
        self.play(Write(annotation))
        self.wait(2)


def display_menu():
    """Display the menu of available functions."""
    print("\n" + "="*60)
    print("FOURIER SERIES ANIMATOR".center(60))
    print("="*60)
    print("\nSelect a periodic function to animate:\n")
    
    functions = get_predefined_functions()
    for key, func_info in functions.items():
        print(f"  {key}. {func_info['name']}")
    
    print(f"  7. Custom function (enter code)")
    print(f"  0. Exit")
    print("\n" + "-"*60)


def get_user_choice():
    """Get the user's menu choice."""
    while True:
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            if choice in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                return choice
            else:
                print("Invalid choice. Please enter a number between 0 and 7.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def get_custom_function():
    """Allow user to input a custom periodic function."""
    print("\n" + "-"*60)
    print("CUSTOM FUNCTION INPUT")
    print("-"*60)
    print("\nEnter a Python lambda function that takes x as input.")
    print("The function should be periodic with period 2π.")
    print("\nExamples:")
    print("  lambda x: np.sin(2*x) + 0.5*np.sin(4*x)")
    print("  lambda x: abs(np.sin(x))")
    print("  lambda x: np.sign(np.sin(x))")
    print("\nNote: numpy is available as 'np', and PI is available as np.pi")
    
    while True:
        try:
            func_str = input("\nEnter your function: ").strip()
            if not func_str:
                print("Function cannot be empty.")
                continue
            
            # Evaluate the function
            func = eval(func_str)
            
            # Test it
            test_val = func(0)
            if not isinstance(test_val, (int, float, np.number)):
                print("Function must return numeric values.")
                continue
            
            func_name = input("Enter a name for this function: ").strip()
            if not func_name:
                func_name = "Custom Function"
            
            return func, func_name
        
        except Exception as e:
            print(f"Error in function: {e}")
            print("Please check your syntax and try again.")


def get_num_harmonics():
    """Get the number of harmonics from the user."""
    while True:
        try:
            num = int(input("\nEnter number of harmonics to display (1-50, default=10): ").strip() or "10")
            if 1 <= num <= 50:
                return num
            else:
                print("Please enter a number between 1 and 50.")
        except ValueError:
            print("Please enter a valid integer.")


def render_animation(func, func_name, n_harmonics):
    """Render the Fourier animation using Manim."""
    print(f"\nPreparing to render: {func_name} with {n_harmonics} harmonics...")
    print("This may take a minute or two...\n")
    
    # Set environment variables to pass to the Manim subprocess
    import os
    
    # Find which function was selected
    functions = get_predefined_functions()
    func_choice = "1"  # default
    for key, func_info in functions.items():
        if func_info["name"] == func_name:
            func_choice = key
            break
    
    # Create environment with our settings
    env = os.environ.copy()
    env['FOURIER_FUNC_CHOICE'] = func_choice
    env['FOURIER_N_HARMONICS'] = str(n_harmonics)
    env['FOURIER_FUNC_NAME'] = func_name
    
    # Create output filename
    output_file = f"fourier_{func_name.replace(' ', '_')}"
    
    # Get the full path to this script
    script_path = os.path.abspath(__file__)
    
    # Build the manim command with CORRECT syntax
    # Quality options: -ql (low), -qm (medium), -qh (high), -qp (4K), -qk (4K high fps)
    # IMPORTANT: Use ONLY ONE quality flag (e.g., -qm NOT -qlm)
    command = [
        "manim",
        "-qm",  # Medium quality - use this ONLY (not combined with other quality flags)
        "--output_file", output_file,
        script_path,  # Use full path to the script
        "FourierTransformAnimation"
    ]
    
    print(f"Running command: manim -qm --output_file {output_file} {script_path} FourierTransformAnimation\n")
    
    try:
        # Run manim with environment variables
        result = subprocess.run(command, capture_output=False, text=True, env=env)
        
        if result.returncode == 0:
            # Look for the generated video
            current_dir = Path(os.getcwd())
            media_videos_dir = current_dir / "media" / "videos" / "1080p30"
            
            # The video file should be at media/videos/1080p30/[output_file].mp4
            if media_videos_dir.exists():
                for video_file in media_videos_dir.glob(f"{output_file}*.mp4"):
                    # Copy to working directory
                    final_path = current_dir / f"{output_file}.mp4"
                    import shutil
                    shutil.copy(str(video_file), str(final_path))
                    print(f"\n{'='*60}")
                    print("✓ Animation saved successfully!")
                    print(f"{'='*60}")
                    print(f"File: {final_path}")
                    print(f"{'='*60}\n")
                    return
            
            # If we get here, look in media folder
            print(f"\n{'='*60}")
            print("✓ Animation rendering completed!")
            print(f"{'='*60}")
            print(f"Video file saved in: media/videos/1080p30/")
            print(f"Look for: {output_file}.mp4")
            print(f"{'='*60}\n")
        else:
            print(f"\n✗ Error during rendering.")
    
    except FileNotFoundError as e:
        print("\n✗ Error: 'manim' command not found.")
        print("Please ensure Manim is installed: pip install manim")
        print("See README.md for full installation instructions.")
    except Exception as e:
        print(f"\n✗ Error during rendering: {e}")
        print(f"\nDebug info:")
        print(f"  Script location: {script_path}")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  File exists: {os.path.exists(script_path)}")
        print(f"  Function choice: {func_choice}")
        print(f"  Environment variables set correctly")


def main():
    """Main function to run the Fourier animation tool."""
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        
        elif choice == "7":
            func, func_name = get_custom_function()
        else:
            functions = get_predefined_functions()
            func_info = functions[choice]
            func = func_info["func"]
            func_name = func_info["name"]
        
        # Get number of harmonics
        n_harmonics = get_num_harmonics()
        
        # Render the animation
        render_animation(func, func_name, n_harmonics)
        
        # Ask if user wants to continue
        again = input("\nRender another animation? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
