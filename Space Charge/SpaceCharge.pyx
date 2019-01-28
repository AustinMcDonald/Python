import numpy as np
cimport numpy as np

cdef double FORCE_NEW(double r):
    cdef double K = 9e9
    cdef double Q = 1.6e-19
    cdef double F
    F = K*Q**2/r**2
    return F

cdef double VELOCITY_NEW(double V0,double F,double t):
    cdef double m=1e-31
    cdef double Vnew
    Vnew = V0 + F*t/m
    return Vnew

cdef double RADIUS_NEW(double Vnew,double t):
    cdef double d
    d = Vnew*t
    return d

cpdef SPACE_CHARGE(int Nparticles,double DDistance,int NNsteps,double TimeStep):
    
    cdef int Number       = int(Nparticles)
    cdef double Distance  = DDistance
    cdef double Tstep        = TimeStep
    cdef int Nsteps       = NNsteps
    cdef int I
    cdef int II
    cdef int III
    cdef np.ndarray Velocity = np.zeros(Number)
    cdef np.ndarray XplaceS  = np.random.normal(0, Distance, Number)
    Xplace = XplaceS
    cdef np.ndarray Xnew   = np.zeros(Number)
    
    for III in range(0,Nsteps):
        #print("On loop",III)
        for I in range(0,len(Xplace)):
            Xcurrent = Xplace[I]
            FORCE = 0
            for II in range(0,len(Xplace)):
                DeltaX = (Xcurrent - Xplace[II])
                if DeltaX !=0.0:
                    if Xcurrent > Xplace[II]:
                        FORCE +=  FORCE_NEW(DeltaX)
                    else:
                        FORCE += -FORCE_NEW(DeltaX)
			
            Velocity[I] = VELOCITY_NEW(Velocity[I], FORCE, Tstep)
            Xnew[I]     = RADIUS_NEW(Velocity[I], Tstep)
	    
        Xplace = Xnew
        
    return Xplace, XplaceS