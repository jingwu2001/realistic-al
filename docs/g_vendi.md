#### 1. Implement Gradient Sampling
###### Roadmap
- get gradient
- compute kernel matrix
- compute vendi score or q-vendi
###### Questions
- where do we include diversity
###### Pseudo-code (currently not considering acceleration of kernel computation)
```
# Get Gradient
if last_layer:
	if follow_badge_pseudolabel:
		grad_l = get_gradient()
		grad_u = get_gradient() # using the same function from badge, but with ground truth label
grad_l_normed, norm_l = normalize(grad_l) # (N, d)
grad_u_normed, norm_u = normalize(grad_u)

# Compute kernel matrix
# Assuming linear kernel
kernel_L = grad_l_normed.T @ grad_l_normed

for i, grad_u, norm in zip(grad_u_normed, norm_u):
	
	

```
