def newton(g,Dg, x0, epsilon=1e-10, max_iters=500):
    xn = x0
    for n in range(0,max_iters):
        gxn = g(xn)
        if abs(gxn) < epsilon:
            print('Found solution after ',n,' iterations')
            return xn
        Dgxn = Dg(xn)
        if Dgxn == 0:
            print("Zero Derivative. No Solution Found")
            return None
        
        xn = xn - gxn / Dgxn
    
    print('Max Iters Exceeded. No Solution Found')
    return None

if __name__ == '__main__':
    p = lambda x: x**3 - x**2 - 1
    Dp = lambda x: 3*x**2 - 2*x
    approx = newton(p,Dp,1)
    print(approx)