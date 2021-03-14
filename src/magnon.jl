include("ode.jl")
using LinearAlgebra, NPZ

function magnon_integrate(X_0,T_0,T_max,h,f,step)
    N = Int(floor(T_max/h))
    shape = size(X_0)

    # Adding 2 to N for fractional step at end + initial position

    # the spin array is two dimensional 
    
    X = zeros(Float64, (N+2,shape[1],shape[2]))
    T = zeros(Float64, N+2)
    
    t = T_0
    X[1,:] = X_0
    for i ∈ 1:N+1
        h = min(h,T_max - t)
        X[i+1,:] = step(t,X[i,:],h,f)
        T[i+1] = t + h
        t = T[i+1]
    end
    return T, X
end


# Doing the error analysis in julia

mu = 1
B  = [0,0,1]
gamma = 1

function gradH(S,n)
    heff = mu .* B
    H    = zeros(3,n)
    for i in 1:n
        H[:,i] = heff
    end
    return H
end

# for this particular example, the effective field is the same
# every time. Thus we don't need to calculate it every time
# also, n == 1

dH = gradH(nothing,1)
C = - gamma/mu
n = 1

function f_llg(t,S)
    return C * ( cross(S[:,1],dH[:,1]) )
end

function initial_cond(theta,phi)
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)

    return [x,y,z]
end

function S_xy(wt,a,b)
    x = a * cos(wt) - b * sin(wt)
    y = b * cos(wt) + a * sin(wt)

    return [a,b]
end

function error_analysis(N)

    tN  = 2 * pi
    hs  = exp10.(range(-5,-1, length=N)) # logspace
    
    S_0 = initial_cond(0.1,0.5)
    S_0 = reshape(S_0,(3,1))
    S_a = S_xy(tN,S_0[1,1],S_0[2,1]) # analytical sol at endpoint

    errs = zeros(2,2,N) # global errors
    times = zeros(2,N)  # runtimes

    for (i,h_i) in enumerate(hs)
        tic = time()
        _, X_heun = magnon_integrate(S_0,0,tN,h_i,f_llg,heun)
        toc = time()

        times[1,i] = toc - tic
        
        tic = time()
        _, X_euler = magnon_integrate(S_0,0,tN,h_i,f_llg,euler)
        toc = time()

        times[2,i] = toc - tic

        errs[1,1,i] = abs(S_a[1] - X_heun[end,1])
        errs[1,2,i] = abs(S_a[2] - X_heun[end,2])

        errs[2,1,i] = abs(S_a[1] - X_euler[end,1])
        errs[2,2,i] = abs(S_a[2] - X_euler[end,2])
    end
    npzwrite("../data/hs_j.npy",hs)
    npzwrite("../data/errs_j.npy",errs)
    npzwrite("../data/times_j.npy",times)
    
end
