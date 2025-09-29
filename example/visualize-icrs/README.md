Visualization for Image-Centered Ray Shooting (ICRS)
================================

Visual aid for the ICRS. The script traces the source limb 
through the lens, builds the polar integration regions, and
plots the resulting sampling tiles alongside the lens caustics.

Run::

    python visualize_paper.py

This generates `visualize_example.png`. The script enables double precision in
JAX and depends on Matplotlib and Seaborn for styling.
