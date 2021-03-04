# integrators in julia

function heun(t,x,h,f)
    k1 = f(t    ,x)
    k2 = f(t + h, x + k1*h)
    x_ = x .+ h*(k1+k2)/2
    return x_
end

function euler(t,x,h,f)
    k1 = f(t,x)
    x_ = x .+ h*k1
    return x_
end

function rk4(t,x,h,f)
    k1  = f( t      , x           )
    k2  = f( t + h/2, x + k1 * h/2)
    k3  = f( t + h/2, x + k2 * h/2)
    k4  = f( t + h  , x + k3 * h  )

    x_ = x .+ h .*(k1 .+ 2 .* k2 .+ 2 .*k3 .+ k4) / 6
    return x_
end

function integrate(X_0,T_0,T_max,h,f,step)
    N = Int(T_max/h)
    shape = size(X_0)[1]

    # Adding 2 to N for fractional step at end + initial position
    
    X = zeros(Float64, (N+2,shape))
    T = zeros(Float64, N+2)
    
    t = T_0
    X[1,:] = X_0
    for i ∈ 1:N+1
        h = min(h,Tmax - t)
        X[i+1,:] = step(t,X[i,:],h,f)
        T[i+1] = t + h
        t = T[i+1]
    end
    return T, X
end

function magnon_integrate(X_0,T_0,T_max,h,f,step)
    N = Int(T_max/h)
    shape = size(X_0)

    # Adding 2 to N for fractional step at end + initial position

    # the spin array is two dimensional 
    
    X = zeros(Float64, (N+2,shape[1],shape[2]))
    T = zeros(Float64, N+2)
    
    t = T_0
    X[1,:] = X_0
    for i ∈ 1:N+1
        h = min(h,Tmax - t)
        X[i+1,:] = step(t,X[i,:],h,f)
        T[i+1] = t + h
        t = T[i+1]
    end
    return T, X
end
