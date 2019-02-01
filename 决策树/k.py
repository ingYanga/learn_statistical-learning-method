import numpy as np
from DrawTools import DrawTools

def main():
    dt = DrawTools()
    def func(x):
        y = -x * np.log(x)
        return y
    
    dt.draw_line_with_func(func, step=0.01, x_range=(0, 1))

if __name__ == "__main__":
    main()

    