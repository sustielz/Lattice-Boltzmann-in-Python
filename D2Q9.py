from lattice import Lattice

import numpy as np;
import matplotlib.pyplot as plt;




class D2Q9(Lattice):
    dct = [(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)]; #Returns the two values corresponding to the direction
    opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]; #returns the opposite direction
    w = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]; #weighting function
    
    
    def __init__(self, N, M):
        super(D2Q9, self).__init__([N, M])
          
    @property
    def N(self): return self.shape[0]
    
    @property
    def M(self): return self.shape[1]
        
    @property
    def vx(self): return self.V[0]
    
    @property
    def vy(self): return self.V[1] 
    
    @property
    def Fx(self): return self.Frce[0]
    
    @property
    def Fy(self): return self.Frce[1]
        
        
####    Boundary-specific methods   ####
    def setupWalls(self):
        N, M = self.shape
        self.wall = np.zeros([self.ndct, *(self.shape+2)])  #### 9xNxM mask ==1 if (i, j) is on wall I; 0 otherwise
        self.WALL = np.zeros([self.ndct, *(self.shape+2)])  #### 9xNxM mask ==1 iff (i,j) on any wall and (i+ex, j+ey) in-bounds (non-periodic)
        
        allowable = lambda I, dct: all([self.dct[I][q]==0 or dct[q] != self.dct[I][q] for q in range(len(self.shape))])
        self.WALL_DCT = [[I for I in range(9) if allowable(I, dct)] for dct in self.dct]
        self.WALL_DCT[0] = []
        
        xind = [list(range(N)), [N], [-1]] ##walls at x=-1, x=N, y=-1, y=M
        yind = [list(range(M)), [M], [-1]]       
        for I in range(1, self.ndct):
            ex, ey = self.dct[I]
            self.wall[I, xind[ex], yind[ey]] = 1
            
        self.walls = np.sum(self.wall, axis=0)
        for I in range(1, 9):
            for J in self.WALL_DCT[I]:
                self.WALL[J] += self.wall[I]
        self.NOWALL = [self.walls - self.wall[I] for I in range(9)]                                   
        
        
    
    #### Boundary Method for a No-Slip Channel
    def set_pois(self, P1, P2): 
        self.boundary = lambda: self.pois(P1, P2)
        self.F_top = [0.0, 0.0]
        self.F_bot = [0.0, 0.0]
        self.X = self.h*(-0.5 + np.arange(self.N+2))
        self.Y = self.h*np.arange(self.M+2)

    
    def pois(self, P1, P2): 
        N, M = self.N, self.M;
        X = np.arange(-1, N+1)
        
        ##Bounce on top and bottom walls (i.e. no-slip)        
        for I in [2, 5, 6]:
            ex, ey = self.dct[I];
##            print self.dct[I], ex, ey
            J = self.opp[I];
            self.f[I][X, -1] = self.fs[J][X, -1];
            self.f[J][X, M] = self.fs[I][X, M];
                
            self.F_bot[0] += ex*(self.fs[I][X, -1] + self.f[J][X, -1]);
            self.F_bot[1] += ey*(self.fs[I][X, -1] + self.f[J][X, -1]);
            self.F_top[0] += ex*(self.fs[J][X, M] + self.f[I][X, M]);
            self.F_top[1] += ey*(self.fs[J][X, M] + self.f[I][X, M]);
##                print np.size(self.F_bot)
##                dP_top =  self.f[J][i, M] + self.fs[I][i, M];
##                self.F_top += ex*dP_top, ey*dP_top;
##            print self.f[2][:, -1]
##            print self.f[4][:, -1]
##            
##            print self.f[5][:, -1]
##            print self.f[6][:, -1]
##        self.bounce(self.corners);  ##Bounce corners
####        ###Zou-He no-slip walls
####        for i in range(N):
####            self.f[2][i, -1] = self.f[4][i, -1];
####            self.f[5][i, -1] = self.f[7][i, -1] - 0.5*(self.f[1][i, -1] - self.f[3][i, -1]);
####            self.f[6][i, -1] = self.f[8][i, -1] + 0.5*(self.f[1][i, -1] - self.f[3][i, -1]);
####
####            self.f[4][i, M] = self.f[2][i, M];
####            self.f[7][i, M] = self.f[5][i, M] - 0.5*(self.f[1][i, -1] - self.f[3][i, -1]);
####            self.f[8][i, M] = self.f[6][i, M] + 0.5*(self.f[1][i, -1] - self.f[3][i, -1]);
####        
####
        ######Zou-He boundary conditions to simulate pressure gradient at sides (copied from zou-he paper)
        rhoL = P1/self.cs**2;
        rhoR = P2/self.cs**2;
        vyL = 0.0;
        vyR = 0.0;

        fL = np.zeros([9, M+2]);
        fR = np.zeros([9, M+2]);

        Y = np.arange(-1, M+1)
        for I in range(9):
            fL[I] = self.f[I][-1, Y]
            fR[I] = self.f[I][N, Y]

        ##Left wall
        vxL = 1.0 - (1/rhoL)*(fL[0] + fL[2] + fL[4] + 2.0*(fL[3] + fL[6] + fL[7]));
        self.f[1][-1, Y] = fL[3] + (2.0/3.0)*rhoL*vxL    
        self.f[5][-1, Y] = fL[7] - 0.5*(fL[2] - fL[4]) + (1.0/6.0)*rhoL*vxL + 0.5*rhoL*vyL;
        self.f[8][-1, Y] = fL[6] + 0.5*(fL[2] - fL[4]) + (1.0/6.0)*rhoL*vxL - 0.5*rhoL*vyL;

        ##Right wall
        vxR = -1.0 + (1/rhoR)*(fR[0] + fR[2] + fR[4] + 2.0*(fR[1] + fR[5] + fR[8])) ;
        self.f[3][N, Y] = fR[1] - (2.0/3.0)*rhoR*vxR    
        self.f[7][N, Y] = fR[5] + 0.5*(fR[2] - fR[4]) - (1.0/6.0)*rhoR*vxR - 0.5*rhoR*vyR;
        self.f[6][N, Y] = fR[8] - 0.5*(fR[2] - fR[4]) - (1.0/6.0)*rhoR*vxR + 0.5*rhoR*vyR;

        
        

        
    #########   Methods for visualizing the lattice    ###############
    def im(self, dat): return np.transpose(np.roll(dat, (1, 1), axis=(0, 1)))
    
    def imshow(self, dat): 
        center = lambda mask: np.roll(mask, (1, 1), axis=(0, 1))
        plt.imshow(self.im(dat), origin='lower', cmap='hot')            
        plt.colorbar()
        
    def imshow9(self, DAT, title=None, middle=False):
        II = range(9) if middle else range(1, 9)
        for I in II:
            ex, ey = self.dct[I]
            plt.subplot(3, 3, (ex+2) + 3*(-ey+2-1))
            self.imshow(DAT[I])
            plt.title('direction {}: ({}, {})'.format(I, ex, ey))
        plt.suptitle(title)
        plt.tight_layout()
#         plt.show()
        
    def displayWalls(self): 
        self.imshow9(self.wall + 0.5*self.WALL, 'Display Wall Protocol')
        
    def displayBounce(self): 
        mask = [self.SOLID[I] + 0.1*self.SOLID[I][self.DCT[self.opp[I]]] for I in range(9)]
        self.imshow9(mask, 'Bounce Protocol')
        plt.subplot(3, 3, 5)
        self.imshow(self.solid)
        plt.title('Immersed Solids')
        
    def displayStream(self): 
        mask = [0.5*self.f[I][self.DCT[I]] +  self.fs[I] + 0.1*(self.walls-self.WALL[I]) + 0.2*self.SOLID[I][self.DCT[self.opp[I]]] for I in range(9)]
        self.imshow9(mask, 'Streaming')    

    def plotF(self): self.imshow9(self.f, 'f', True)           
    def plotFs(self): self.imshow9(self.fs, 'fs', True)           
        
        
    ########## Methods for visualizing macroscopic flow  #####
    def plotFlow(self):
        h = self.h;
        X = self.X;
        Y = self.Y;
        U = (self.h/self.s)*self.im(self.vx)
        V = (self.h/self.s)*self.im(self.vy)
        plt.streamplot(X, Y, U, V, density=1.4, color='black');
        plt.ylim(0, h*self.M);
        plt.xlim(0, h*self.N);

##        plot_Circle(30, self.M/2 - 1, self.h**2, 15);
        plt.imshow(self.im(1-(self.solid==1)),  extent=[0, self.L, 0, self.L*float(self.M)/self.N], alpha=0.5, cmap='gray', origin="lower", aspect='auto');
        plt.gca().set_aspect('equal');

    def plotVorticity(self):
        h = self.h;
        X = self.X;
        Y = self.Y;
        U = (self.h/self.s)*self.vx;
        V = (self.h/self.s)*self.vy;
        omega = np.ones([self.N, self.M]);
        for i in range(-1, self.N-1):
                for j in range(-1, self.M-1):
                    omega[i+1, j+1] = (0.5/h)*( U[i+1, j] - V[i-1, j] - (V[i, j+1] - U[i, j-1]) );
        plt.imshow(np.transpose(omega), origin='lower', interpolation='nearest', extent=[0, self.L, 0, self.L*float(self.M)/self.N]);
        plt.colorbar();
        return omega;

#     def plotProfile(self):
#         h = self.h;
#         N, M, L = self.N, self.M, self.L;
#         H = (M+1)*float(L)/(N+1);
#         cp = self.h/self.s;
#         Yvals = np.arange(-1, M+1);
        
#         X, Y = self.X, self.Y;
        
#         U = self.vx;
#         plt.plot(Y, U[N/2, :][Yvals], label="vx profile");
#         EXACT =  0.5*(self.cs**2)*(self.P2 - self.P1)*Y*(Y - H)/(self.eta) ;
# ##        EXACT = ( (self.h/self.s)*(self.P2 - self.P1)*(self.cs**2)/(2*self.eta) )*(self.L**2-4Y**2);
#         plt.plot(Y, EXACT, label="exact sol'n");
#         plt.title("t = {}".format(self.t));
#         plt.xlabel("y");
#         plt.ylabel("v_x");
        
#         plt.legend();
#         return np.sqrt(sum( (U[N/2, :][Yvals] - EXACT)**2 ));
    
    