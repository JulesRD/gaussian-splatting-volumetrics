# This is where volumetrics live.
#
# Responsibilities:
#
# Alpha compositing
#
# Surface-first vs volume-first blending
#
# Depth-aware accumulation
#
# Example structure:
#
# def composite(surface_rgba, volume_rgba):
#     # front-to-back alpha blending
#     return final_image
#
#
# This isolation makes experimentation easy.