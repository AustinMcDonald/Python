import numpy as np
    
def FORCE_NEW(double r):
    cdef double K = 9e9
    cdef double Q = 1.6e-19
    F = K*Q**2/r**2
    return F

def VELOCITY_NEW(double V0,double F,double t):
    cdef double m=1e-31
    Vnew = V0 + F*t/m
    return Vnew

def RADIUS_NEW(double Vnew,double t):
    d = Vnew*t
    return d

def SPACE_CHARGE(int Nparticles,double DDistance,int NNsteps,double TimeStep):

    cdef int Number       = Nparticles
    cdef double Distance  = DDistance
    cdef double Tstep     = TimeStep
    cdef int Nsteps       = NNsteps
    cdef int I
    cdef int II
    cdef int III
    cdef int hold = 10

    Velocity = np.zeros(Number)
    XplaceS  = np.random.normal(0, Distance, Number)

    Xplace = XplaceS
    Xnew   = np.zeros(Number)

    for III in range(0,Nsteps):
        if III % hold == 0:
            print("On loop",III)
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