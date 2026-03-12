from manim import *
from scipy.integrate import quad
import numpy as np
from typing import Callable, Tuple, List
import sys
import os
from pathlib import Path
import subprocess


class FourierSeriesCalculator:
    
    def __init__(self, func: Callable, period: float = 2 * PI):
        self.func = func
        self.period = period
    
    def calculate_fourier_coefficients(self, n_terms: int) -> Tuple[float, List[float], List[float]]:
        a0_integral, _ = quad(self.func, 0, self.period)
        a0 = (2 / self.period) * a0_integral
        
        a_coefficients = []
        b_coefficients = []
        
        for n in range(1, n_terms + 1):
            def an_integrand(x):
                return self.func(x) * np.cos(2 * n * np.pi * x / self.period)
            
            an_integral, _ = quad(an_integrand, 0, self.period)
            an = (2 / self.period) * an_integral
            a_coefficients.append(an)
            
            def bn_integrand(x):
                return self.func(x) * np.sin(2 * n * np.pi * x / self.period)
            
            bn_integral, _ = quad(bn_integrand, 0, self.period)
            bn = (2 / self.period) * bn_integral
            b_coefficients.append(bn)
        
        return a0, a_coefficients, b_coefficients
    
    def fourier_approximation(self, x: float, a0: float, a_coeffs: List[float], b_coeffs: List[float]) -> float:
        result = a0 / 2
        for n, (an, bn) in enumerate(zip(a_coeffs, b_coeffs), 1):
            result += an * np.cos(2 * n * np.pi * x / self.period) + \
                      bn * np.sin(2 * n * np.pi * x / self.period)
        return result


def get_predefined_functions():
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
    
    def construct(self):
        self.camera.background_color = WHITE
        
        import os
        func_choice = os.environ.get('FOURIER_FUNC_CHOICE', '1')
        n_harmonics = int(os.environ.get('FOURIER_N_HARMONICS', '10'))
        function_name = os.environ.get('FOURIER_FUNC_NAME', 'Function')
        
        functions = get_predefined_functions()
        
        if func_choice in functions:
            func = functions[func_choice]["func"]
            function_name = functions[func_choice]["name"]
        else:
            func = functions["1"]["func"]
            function_name = functions["1"]["name"]
        
        period = 2 * PI
        
        calculator = FourierSeriesCalculator(func, period)
        a0, a_coeffs, b_coeffs = calculator.calculate_fourier_coefficients(n_harmonics)
        
        axes_original = Axes(
            x_range=[0, 2 * PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": GREY_B},
            tips=False,
        )
        axes_original.scale(0.6)
        
        max_harmonics = max(50, n_harmonics)
        axes_spectrum = Axes(
            x_range=[0, max_harmonics, 5],
            y_range=[0, 1.2, 0.2],
            axis_config={"color": GREY_B},
            tips=False,
        )
        axes_spectrum.scale(0.6)
        
        axes_original.to_edge(LEFT)
        axes_spectrum.to_edge(RIGHT)
        
        label_original = Text("Original Function", font_size=24, color=BLACK)
        label_original.next_to(axes_original, UP)
        
        label_spectrum = Text("Frequency Spectrum", font_size=24, color=BLACK)
        label_spectrum.next_to(axes_spectrum, UP)
        
        original_func = axes_original.plot(func, color=BLUE, stroke_width=2.5)
        
        self.add(axes_original, axes_spectrum, label_original, label_spectrum)
        self.play(Create(original_func))
        self.wait(1)
        
        fourier_curves = []
        spectrum_bars = []
        
        for harmonic in range(1, n_harmonics + 1):
            current_a_coeffs = a_coeffs[:harmonic]
            current_b_coeffs = b_coeffs[:harmonic]
            
            def fourier_func(x):
                return calculator.fourier_approximation(x, a0, current_a_coeffs, current_b_coeffs)
            
            fourier_curve = axes_original.plot(fourier_func, color=RED, stroke_width=2.5)
            
            magnitude = np.sqrt(a_coeffs[harmonic-1]**2 + b_coeffs[harmonic-1]**2)
            
            bar_height = magnitude / 2
            
            bar = Rectangle(
                width=0.3,
                height=0.5,
                color=GREEN,
                fill_opacity=0.7,
            )
            bar.scale(bar_height / 0.5)
            bar.move_to(axes_spectrum.coords_to_point(harmonic, bar_height / 2))
            
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
        
        annotation = Text(
            f"{function_name} - Fourier Series with {n_harmonics} Harmonics",
            font_size=20,
            color=BLACK
        )
        annotation.to_edge(DOWN)
        self.play(Write(annotation))
        self.wait(2)


def display_menu():
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
            
            safe_dict = {
                'np': np,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'abs': abs,
                'max': max,
                'min': min,
                'pi': np.pi,
            }
            
            func = eval(func_str, {"__builtins__": {}}, safe_dict)
            
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
    print(f"\nPreparing to render: {func_name} with {n_harmonics} harmonics...")
    print("This may take a minute or two...\n")
    
    import os
    
    functions = get_predefined_functions()
    func_choice = "1"
    for key, func_info in functions.items():
        if func_info["name"] == func_name:
            func_choice = key
            break
    
    env = os.environ.copy()
    env['FOURIER_FUNC_CHOICE'] = func_choice
    env['FOURIER_N_HARMONICS'] = str(n_harmonics)
    env['FOURIER_FUNC_NAME'] = func_name
    
    output_file = f"fourier_{func_name.replace(' ', '_')}"
    
    script_path = os.path.abspath(__file__)
    
    command = [
        "manim",
        "-qm",
        "--output_file", output_file,
        script_path,
        "FourierTransformAnimation"
    ]
    
    print(f"Running command: manim -qm --output_file {output_file} {script_path} FourierTransformAnimation\n")
    
    try:
        result = subprocess.run(command, capture_output=False, text=True, env=env)
        
        if result.returncode == 0:
            current_dir = Path(os.getcwd())
            media_videos_dir = current_dir / "media" / "videos" / "1080p30"
            
            if media_videos_dir.exists():
                for video_file in media_videos_dir.glob(f"{output_file}*.mp4"):
                    final_path = current_dir / f"{output_file}.mp4"
                    import shutil
                    shutil.copy(str(video_file), str(final_path))
                    print(f"\n{'='*60}")
                    print("✓ Animation saved successfully!")
                    print(f"{'='*60}")
                    print(f"File: {final_path}")
                    print(f"{'='*60}\n")
                    return
            
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
        
        n_harmonics = get_num_harmonics()
        
        render_animation(func, func_name, n_harmonics)
        
        again = input("\nRender another animation? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
