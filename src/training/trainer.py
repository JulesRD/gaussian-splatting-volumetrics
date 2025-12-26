# High-level loop:
#
# for iter in range(n_iters):
#     render = renderer(scene, camera)
#     loss = compute_loss(render, target)
#     loss.backward()
#     optimizer.step()
#
#
# Keep it generic, so you can:
#
# Freeze surface Gaussians
#
# Optimize only volumetric Gaussians