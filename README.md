FOURIER SERIES ANIMATION
A Python script that creates animated visualizations of periodic functions and their Fourier series decompositions using Manim.
OVERVIEW
This project generates animations showing how periodic functions can be reconstructed from their Fourier series components. Watch as harmonics progressively build up to approximate the original function, with a real-time frequency spectrum display.
FEATURES
Animated Fourier Series: Watch harmonics progressively add up to reconstruct the original function
Frequency Spectrum: Real-time visualization of frequency components
Multiple Functions: 6 predefined periodic functions included
Custom Functions: Support for user-defined periodic functions
Flexible Harmonics: Render with 1-50 harmonics for different levels of detail
MP4 Export: High-quality MP4 video output

DEPENDENCIES
Required Libraries:
-manim
-scipy
-numpy

Python Version: 3.8 or higher

USAGE
Basic Usage:
python fourier_animation.py
This opens an interactive menu where you can:

Select a periodic function (1-6)
Choose the number of harmonics (1-50)
Wait for the animation to render

Interactive Menu:
Select a periodic function to animate:

Square Wave
Sawtooth Wave
Triangle Wave
Sine Wave
Half Sine Wave (Rectified)
Pulse Wave
Custom function (enter code)

Enter your choice (0-7): 
Enter number of harmonics to display (1-50, default=10): 
