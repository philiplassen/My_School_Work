from scipy import io
fname = "data/aut-avn.mat"
content = io.loadmat(fname, struct_as_record=True)
Xt = content['X']
y = content['Y']
print(type(Xt))
print(type(y))
res = y.T @ y
print(res)

