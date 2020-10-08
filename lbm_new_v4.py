import numpy as np;
import matplotlib.pyplot as plt;

###NOTE: I did a time test for a couple of implementations. I.e. how much longer
###       does it take to check whether a node and its neighbors are fluid or solid
#
###Calculated time taken for 40x32 poisselle channel to iterate for a set period of time

###time taken to stream + check node type + check neighbors node type: 27.8, 22.5
###Time taken to stream and check node type: 22.5
###Time taken to stream without checking node type: 19.5
###Time taken if you store a list of which nodes are fluids: 21.9
###I believe it is best to simply use the array. It is important to have functionality for embedded objects.



class D2Q9(object): ##LBM for a D2Q9 lattice. Boundaries at lattice edges must be specified separately. 
    ##Specify parameters specific to a d2q9 lattice
    dct = [(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)]; #Returns the two values corresponding to the direction
    opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]; #returns the opposite direction
    w = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]; #weighting function
    cs = 1.0/np.sqrt(3); ## Speed of sound in lattice units

    ##Default system parameters
    L = 1.0;
    tau = 1.0;
    
    t = 0.0;
    
    Fx = [];     #Force on immersed boundaries
    Fy = [];
    
      

######## Initialization ########
    def __init__(self, N, M):              ##Initialize an NxM lattice, consisting of fluid with walls at edges.
        self.N, self.M = N, M
        indices = lambda dct: np.meshgrid( np.roll(np.arange(N+2), dct[0]), np.roll(np.arange(M+2), dct[1]), indexing='ij' )
        self.DCT = [indices(dct) for dct in self.dct]     #### Coordinates streamed in direction dct[i]
        self.rho = np.zeros([N+2, M+2])   ##walls at -1, N, -1, M
        self.vx = np.zeros([N+2, M+2])
        self.vy = np.zeros([N+2, M+2])
        self.f = np.zeros([9, N+2, M+2])
        self.fs = np.zeros([9, N+2, M+2])
        self.feq = np.zeros([9, N+2, M+2])
        
        
        
#         self.fluid = np.ones([N+2, M+2]); ##Initialize the fluid mask ==1 if fluid, ==0 if solid, ==2 if wall
        
        self.solid = np.zeros([N+2, M+2]) ## solid mask ==1 if solid
        self.setupWalls()
    
    
    @property
    def solid(self): return self._solid       #### NxM mask with 1=solid, 0=fluid
    
    @property
    def SOLID(self): return self._SOLID       #### 9xNxM mask ==1 if (i, j) is solid and (i+ex, j+ey) is fluid; 0 otherwise
    
    @solid.setter
    def solid(self, mask):
        self._solid = mask
        self._SOLID = np.array([mask - mask[self.DCT[i]] for i in range(9)])
        self._SOLID[self._SOLID<0]=0
        
    def addSolid(self, mask):
        solid = self.solid + mask
        solid[solid>1] = 1
        self.solid = solid
    
    def updateSolid(self): self.solid = self.solid   #### To update adjacency map SOLID, just call the setter
        
    def setupWalls(self):
        N, M = self.N, self.M
        self.wall = np.zeros([9, N+2, M+2])  #### 9xNxM mask ==1 if (i, j) is on wall I; 0 otherwise
        self.WALL = np.zeros([9, N+2, M+2])  #### 9xNxM mask ==1 iff (i,j) on any wall and (i+ex, j+ey) in-bounds (non-periodic)
        
        allowable = lambda I, dct: all([self.dct[I][q]==0 or dct[q] != self.dct[I][q] for q in [0, 1]])
        self.WALL_DCT = [[I for I in range(9) if allowable(I, dct)] for dct in self.dct]
        self.WALL_DCT[0] = []
        
        xind = [list(range(N)), [N], [-1]] ##walls at -1, N, -1, M
        yind = [list(range(M)), [M], [-1]]       
        for I in range(1, 9):
            ex, ey = self.dct[I]
            self.wall[I, xind[ex], yind[ey]] = 1
            
        self.walls = np.sum(self.wall, axis=0)
        for I in range(1, 9):
            for J in self.WALL_DCT[I]:
                self.WALL[J] += self.wall[I]
        self.NOWALL = [self.walls - self.wall[I] for I in range(9)]
#         self.displayWalls()
    
#     def setupBounce(self):  #### Initiate initial bounceback conditions to start with zero velocity at solid walls
#         for I in range(9):
#             self.
    

#     def updateFluid(self):
#         N, M = self.N, self.M;
# #         self.fluid = np.ones([N+2, M+2]) ##Initialize the fluid mask ==1 if fluid, ==0 if solid, ==2 if wall
# #         self.fluid -= self.solid
#         self.bounce = np.array([self.solid - self.solid[ self.DCT[i] ] for i in range(9)])
#         self.bounce[self.bounce<0]=0
                       
# #         self.bounce = [self.solid*((1-self.solid)[self.indices[i]]) for i in range(9)]
# #         for i in range(9):
# #             plt.imshow(self.solid - self.solid[self.indices[i]])
# #             plt.colorbar()
# #             plt.show()
                       
#         self.bounce_dest = [np.roll(self.bounce[i], self.dct[self.opp[i]], axis=(0, 1)) for i in range(9)]
        
                       

#         for I in [1, 2, 5, 6]:
#             bi = self.bounce[I]
#             bj = self.bounce[self.opp[I]]
#             double = bi*bj
#             bi += double
#             bj += double

#         self.displayBounce()
#         self.displayBounce(True)
#         self.displayBounce(False)
    
        
   
        
        
        


                                          

######## Methods to update macroscopic quantities and f_eq ########
    def updateRho(self):
#         self.rho = np.sum(self.f, axis=0)
        self.rho = self.rho*0.0;
        for I in range(9):
            self.rho += self.f[I];
##        print self.rho

    def updateV(self):
        self.vx = self.vx*0;
        self.vy = self.vy*0;
        for I in range(9):
            ex, ey = self.dct[I];
            self.vx += (1.0/self.rho)*self.f[I]*ex;
            self.vy += (1.0/self.rho)*self.f[I]*ey;

    def updateEq(self, rho, vx, vy):
        for I in range(9):
            ex, ey = self.dct[I];
            A = ex*vx + ey*vy;
            B = vx*vx + vy*vy;
            self.feq[I] = self.w[I]*rho*( 1 + 3.0*A + 4.5*A**2 - 1.5*B );

###############
#     def Stream(self):        
#         fnew = [self.fs[I][self.DCT[I]] for I in range(9)]   ## Stream everything
# #         self.displayBounce()
#         for I in range(9):           
#             J = self.opp[I]
#             adj = self.SOLID[I]#[self.DCT[I]]                 ## Set up solid bounce
#             opp_adj = self.SOLID[J]#[self.DCT[J]]                 ## Set up solid bounce
# #             fluid = 1 - self.solid + 1.5*self.SOLID[I] - .5*adj - self.walls + self.WALL[J]   ## Set up fluid mask
#             fluid = 1 - self.solid + adj   ## Set up fluid mask
# #             if I==1: 
# #                 plt.clf()
# #                 plt.imshow(np.roll(fluid+2*adj, (1, 1), axis=(0, 1)))
# #                 plt.colorbar()
# #                 plt.show()
#             self.f[I] = fnew[I]*fluid + self.fs[J]*opp_adj#elf.SOLID[I]
    

#     def Collision(self, s):
#         mask = self.solid - np.sum(self.SOLID, axis=0)
#         for I in range(9):  ##Perform Collision on Everything  
#             self.fs[I][mask==0] = (self.f[I] + (1/self.tau)*(self.feq[I] - self.f[I]))[mask==0];
# #             self.fs[I] = (self.f[I] + (1/self.tau)*(self.feq[I] - self.f[I]))
####################

    def Stream(self):        
        fnew = [self.fs[I][self.DCT[I]] for I in range(9)]   ## Stream everything
#         self.displayBounce()
        for I in range(9):           
            fluid = 1 - self.solid + self.SOLID[I]            ## Set up solid mask 
            self.f[I] = fnew[I]*fluid ######+ self.fs[J]*opp_adj#elf.SOLID[I]
    

    def Collision(self, s):
        for I in range(9):      ## Perform (halfway) bounceback on solid edges 
            J = self.opp[I]
            self.fs[I][self.SOLID[J]==1] = self.f[J][self.SOLID[J]==1]
        for I in range(9):
            self.fs[I][self.solid==0] = (self.f[I] + (1/self.tau)*(self.feq[I] - self.f[I]))[self.solid==0]
#             self.fs[I] = (self.f[I] + (1/self.tau)*(self.feq[I] - self.f[I]))

    def boundary(self): ##In general, the boundary will depend on the system
        pass;
    
#     def bounce_walls(self):
#         for I in range(9):
#             J = self.opp[I]
#             mask = (self.walls-self.WALL[I])==1
# #             plt.imshow(np.roll(self.f[I]*mask[self.DCT[J]], (1, 1), axis=(0, 1)))
# #             plt.colorbar()
# #             plt.show()
# #             self.f[J] += self.fs[I][self.DCT[I]]*mask 


# #             self.f[J][mask] = self.f[I][mask] 
# #             self.f[J] = self.f[I]*mask 
    
    def iterate(self, s):
        self.updateRho()
        self.updateV()
        self.updateEq(self.rho, self.vx, self.vy)
        self.Collision(s)
        self.Stream()
        self.boundary()
        
        self.t += s
    
            
            
    def imshow(self, dat): 
        center = lambda mask: np.roll(mask, (1, 1), axis=(0, 1))
        plt.imshow(np.transpose(center(dat)), origin='lower', cmap='hot')            
        plt.colorbar()
        
    def imshow9(self, DAT, title=None, middle=False):
#         center = lambda mask: mask[self.DCT[5]]
        II = range(9) if middle else range(1, 9)
        for I in II:
            ex, ey = self.dct[I]
            plt.subplot(3, 3, (ex+2) + 3*(-ey+2-1))
            self.imshow(DAT[I])
            plt.title('direction {}: ({}, {})'.format(I, ex, ey))
        plt.tight_layout()
        plt.suptitle(title)
#         plt.show()
        
    def displayWalls(self): 
        self.imshow9(self.wall + 0.5*self.WALL, 'Display Wall Protocol')
        
    def displayBounce(self): 
        mask = [self.SOLID[I] + 0.1*self.SOLID[I][self.DCT[self.opp[I]]] for I in range(9)]
        self.imshow9(mask, 'Display Bounce Protocol')
        
    def displayStream(self): 
        mask = [0.5*self.f[I][self.DCT[I]] +  self.fs[I] + 0.1*(self.walls-self.WALL[I]) + 0.2*self.SOLID[I][self.DCT[self.opp[I]]] for I in range(9)]
        self.imshow9(mask, 'Streaming')    

        
    
        
#     def displayStream(self, dest=True):
#         center = lambda mask: np.roll(mask, (1, 1), axis=(0, 1))
# #         plt.subplot(3, 3, 5)
# #         plt.imshow(center(self.walls))
# #         plt.title('All Walls')
#         for I in range(9):
#             ex, ey = self.dct[I]
#             plt.subplot(3, 3, (ey+2) + 3*(ex+2-1))
#             plt.title('direction {}: ({}, {})'.format(I, ex, ey))
#             if dest:
# #                 plt.imshow(0.5*self.f[I][self.DCT[I]] +  self.fs[I]+ 0.1*self.solid + 0.2*self.SOLID[I][self.DCT[self.opp[I]]])
#                 plt.imshow(center(0.5*self.f[I][self.DCT[I]] +  self.fs[I] + 0.1*(self.walls-self.WALL[I]) + 0.2*self.SOLID[I][self.DCT[self.opp[I]]]))
#             else:
#                 plt.imshow(self.fs[I] + 0.1*self.solid)
#             plt.colorbar()
#         plt.suptitle('Display Wall Protocol')
#         plt.show()
        
    def getImage(self, prop):
        N, M = self.N, self.M;
        im = np.zeros([N+2, M+2]);
        for i in range(0, N+2):
            for j in range(0, M+2):
                im[i, j] = prop[i-1,j-1];

        return np.transpose(im);                
                       
    def plotF(self):
        pos = [5, 6, 2, 4, 8, 3, 1, 7, 9]; 
        for k in range(3):
            for l in range(3):
                n = 3*k + l;
                
                plt.subplot(3, 3, pos[n]);
                plt.title("f_{}".format(n));
                plt.imshow(self.getImage(self.f[n]), cmap='hot', origin='lower', interpolation='nearest');
                plt.colorbar()
        plt.show();

    def plotFs(self):
        pos = [5, 6, 2, 4, 8, 3, 1, 7, 9];
        for k in range(3):
            for l in range(3):
                n = 3*k + l;
                plt.subplot(3, 3, pos[n]);
                plt.title("f_{}".format(n));
                plt.imshow(self.getImage(self.fs[n]), cmap='hot',  origin='lower', interpolation='nearest')
                plt.colorbar()
        plt.show()   

if __name__ == '__main__':
    
    #### First: Let's construct a lattice to test just the streaming
    test = D2Q9(8, 8)
#     test.tau = 1e8 #### This makes collision operator essentially f = fs; i.e. no collision
    test.f[:, 5, 5] = 1          #### Instantiate f=1 at center in all directions
    test.f[:, 6, 2] = 1
#     for I in range(9):
#         test.f[I][3, 3] = 1    
#         test.f[I][1, 5] = 1    #### Include a point that's next to the wall
#         test.f[I][6, 0] = 1    #### Include a point that's on the wall
    f0 = test.f.copy()
    
#     #### Stream everything
#     for i in range(7):
#         test.fs = test.f
#         test.start_Stream()
#         test.stream_Fluid(nowalls=False)
    
#     #### Stream everything except walls
#     test.f = f0.copy()
#     for i in range(7):
#         test.fs = test.f
#         test.start_Stream()
#         test.stream_Fluid()
        
        
#    #### Stream everything, but stream walls seperately (i.e. non-periodic)
#     test.f = f0.copy()
#     for i in range(7):
#         test.fs = test.f
#         test.start_Stream()
#         test.stream_Fluid()
#         test.stream_Walls()
        
    test.f = f0.copy()
#     test.solid[2:4, 2:4] = 1
    test.updateSolid()
    for i in range(7):
        test.fs = test.f
        test.displayStream()
        test.Stream()
#         test.stream_Fluid()
#         test.stream_Walls()
#         test.bounce_Solid()
#         test.boundary()
        test.bounce_walls()
#         test.boundary()
#         test.Bounce(self.solid)
#         test.Bounce(self.solid)
        
    
    
#     test.f = f0.copy()
#     test.solid[2:4, 2:4] = 1
#     test.updateSolid()
# #     test.updateFluid()
#     test.displayBounce(True)
                       
                       
#     for i in range(7):
#         test.fs = test.f
#         test.tryStream()
    
    
    
    
    
#         plt.displayWalls()
#         plt.displaySolid()    
#     test = D2Q9(5, 5);
#     for I in range(9):
#         for i in range(7):
#             for j in range(7):
#                 test.f[I][i-1, j-1] = 7*i + j;
#                 if(i<5 and j<5):
#                     test.fs[I][i, j] = 50*(j+1) + 5*i ;
#                 else:
#                     test.fs[I][i, j] = 8
#                 test.f[I] = 1.0*test.fs[I];

#     s = 1;
#     test.plotF();
#     while(test.t<3*s):
#         print(self.fluid[
#         test.iterate(s);
#         test.plotFs();
#         test.plotF();

##    N = 7;
##    M = 7;
##    test = np.ones([N, M]);
##    solid = np.ones([N, M]);
##    for i in range(2):
##        for j in range(2):
##            solid[i+3,j+3] = 0;
##    test = test*solid;
##    plt.imshow(test);
##    plt.show();
    print("You ran the wrong file, but you 're still doing a great job bud")


