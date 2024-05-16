# neural-network-example
example of general neural network: data, model, code, environment
pip install tqdm


TODO:


## use NN to determine dot array for given picture


the input should be a boundary picture with binary entry and fixed size

the output contain two part: first, whethe a pixel is a dot or not; second, whether two dots are connected.

This should be two NN. could refer to the linear regression


determine whether two dots are connected. But it is not needed in real case.

This should contain in loss function

n_dots_only_in_line

loss function = (n1 +n2)/n_dots_overlapping


for all pixels next to the dot,
minimize 


list of conditions:
- the dots must be on the boundary
- the dots are evenly distant with each other.
- how to adjust the dot density?



