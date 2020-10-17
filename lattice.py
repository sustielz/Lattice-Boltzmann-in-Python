import numpy as np;
import matplotlib.pyplot as plt;


########   This is a vectorized version of the code; i.e. operations are performed using matrices and masks rather than
########   one-point-at-a-time. This should generally be at least slightly faster. 

class Lattice(object): ##LBM for a D2Q9 lattice. Boundaries at lattice edges must be specified separately. 
    ##Specify parameters specific to a d2q9 lattice
    dct= [(0,0)]; #Returns the two values corresponding to the direction
    opp = [0]; #returns the opposite direction
    w = [1.0]; #weighting function
    cs = 1.0/np.sqrt(3); ## Speed of sound in lattice units

    ##Default system parameters
    L = 1.0;
    tau = 1.0;
    t = 0.0;      

######## Initialization ########
    def __init__(self, shape, eta=1.):              ##Initialize an NxM lattice, consisting of fluid with walls at edges.
        self.shape = np.array(shape)
        self.ndct = len(self.dct)
        
        indices = lambda dct: np.meshgrid( *[np.roll(np.arange(shape[i]+2), dct[i]) for i in range(len(shape))], indexing='ij' )
        
        self.DCT = [indices(dct) for dct in self.dct]     #### Coordinates streamed in direction dct[i]
        self.rho = np.zeros(self.shape+2)   ##walls at -1, N; etc
        self.V = np.zeros([len(shape), *(self.shape+2)])
        self.f = np.zeros([self.ndct, *(self.shape+2)])
        self.fs = np.zeros([self.ndct, *(self.shape+2)])
        self.feq = np.zeros([self.ndct, *(self.shape+2)])
        
        self.Frce = [ [] for i in range(len(shape))]

        self.solid = np.zeros(self.shape+2) ## solid mask ==1 if solid
        self.setupWalls()
        self.eta = eta
     
    
#### Represent immersed solid objects and set up boundary masks
    @property
    def solid(self): return 1*self._solid       #### NxM mask with 1=solid, 0=fluid
    
    @property
    def SOLID(self): return 1*self._SOLID       #### 9xNxM mask ==1 if (i, j) is solid and (i+ex, j+ey) is fluid; 0 otherwise
    
    @solid.setter
    def solid(self, mask):
        self._solid = mask
        self._SOLID = np.array([mask - mask[self.DCT[i]] for i in range(self.ndct)])
        self._SOLID[self._SOLID<0]=0
        
    def addSolid(self, mask):
        solid = self.solid + mask
        solid[solid>1] = 1
        self.solid = solid
    
    def updateSolid(self): self.solid = self.solid   #### To update adjacency map SOLID, just call the setter
    
    def setupWalls(self): pass                       #### Override this to setup wall masks in each particular lattice                   

######## Configure and update macroscopic quantities and f_eq ########
    @property
    def h(self): return self.L/float(self.N)
    
    @property
    def s(self): return self._s
    
    @property
    def eta(self): return self._eta
    
    @eta.setter
    def eta(self, eta): 
        self._eta = eta
        self._s = (self.tau - 0.5)*(self.h**2)*(self.cs**2)/eta;

#     def set_reynolds(self, Re, v, L):
#         self.set_viscosity(v*L/Re);
     
    def updateRho(self):
        self.rho = np.sum(self.f, axis=0)
        
    def updateV(self):
        self.V *= 0
        for I in range(self.ndct):
            for i, ei in enumerate(self.dct[I]):
                self.V[i] += (1.0/self.rho)*self.f[I]*ei
            
    def updateEq(self, rho, V):
        for I in range(self.ndct):
            A = 0
            B = 0
            for i, ei in enumerate(self.dct[I]):
                A += ei*V[i]
                B += V[i]**2
            self.feq[I] = self.w[I]*rho*( 1 + 3.0*A + 4.5*A**2 - 1.5*B )

#### Stream fluid in all direction. For each direction I, this includes streaming into the incident solid boundary SOLID[I]; and out off the opposite solid boundary SOLID[J]. ## (or is it SOLID[J]; SOLID[I]?)
    def Stream(self):        
        fnew = [self.fs[I][self.DCT[I]] for I in range(self.ndct)]   ## Compute streaming for everything
        for I in range(self.ndct):           
            fluid = 1 - self.solid + self.SOLID[I]           ## Mask for streaming in direction I. 
            self.f[I] = fnew[I]*fluid                        ## Stream with applied mask
    
    def Bounce(self, SOLID):
        for Fi in self.Frce:
            Fi.append(np.zeros(self.shape+2))
        for I in range(self.ndct):
            J = self.opp[I]
            self.fs[I][SOLID[J]==1] = self.f[J][SOLID[J]==1]
            for i, di in enumerate(self.dct[I]):
                self.Frce[i][-1] += self.f[J]*SOLID[J]*di;
    
#     def iBounce(self, solid, i):
#         j = self.opp[i]
#         self.fs[i][solid==1] = self.f[j][solid[j]==1]
#         force = []
# #         for i, di in enumerate(self.dct[i]):
# #             force.append(self.f[j]*solid*di)
# #         return force
#         return [self.f[j]*solid*di for di in self.dct[i]]

    def Collision(self):
        for I in range(self.ndct):
            self.fs[I][self.solid==0] = (self.f[I] + (1/self.tau)*(self.feq[I] - self.f[I]))[self.solid==0]
#             self.fs[I] = (self.f[I] + (1/self.tau)*(self.feq[I] - self.f[I]))

    def boundary(self): ##In general, the boundary will depend on the system
        pass
    
    def iterate(self):
        self.updateRho()      ## f --> rho
        self.updateV()        ## f --> v
        self.updateEq(self.rho, self.V)  ## rho, v --> feq
        self.Bounce(self.SOLID) 
        self.Collision()     ## feq --> fs 
        self.Stream()         ## fs --> f
        self.boundary()       
        self.t += self.s
    