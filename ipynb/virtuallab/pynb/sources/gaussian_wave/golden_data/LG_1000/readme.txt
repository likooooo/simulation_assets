OneDrive\Documents\virtuallab\pynb\sources\gaussian_wave\golden_data\LG_1000 是 virtuallab 生成的 laguerre gauss 光束的结果。 

wavelength   = 0.5
start  = [-5, -5] # 'um'
end    = [5, 5]
shape  = [100, 100]

step   = [(b-a)/(N-1) for a,b,N in zip(start, end, shape)]
meta   = {
    "dx" : step[0] * 1e-3, # set display unit to 'mm'
    "dy" : step[1] * 1e-3,
    "nx" : shape[0],
    "ny" : shape[1]
}
test = [
    [3, 3, 1.3963], [5, 7,  1.3963],[5,7,0], [5,5, 1.3], [5,6,0.7], [5,6, 1.4], 
    [5,6, 1.8], # 超出瑞丽范围之后, 误差显著增大
    [5,6, 1.4784]
]