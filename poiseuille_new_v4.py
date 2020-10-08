import numpy as np;
import matplotlib.pyplot as plt;
import lbm_new_v4 as LBM;
import time as time;
import matplotlib.animation as animation

##It works pretty good. As 0f 12/12/18, final version. To be used with lbm_hybrid


##pressure driven flow for D2Q9 lattice
class poiseuille(LBM.D2Q9): 
    s = 0.0;    ##initialize timestep
    h = 0.0;    ##initialize dx
    F_top = [0.0, 0.0];
    F_bot = [0.0, 0.0];

    VX = [];
    VY = [];
    def __init__(self, N, M, P1, P2, L, eta):
        super(poiseuille, self).__init__(N, M);
        self.P1 = P1;                   ##Pressure at inlet (1) and outlet (2)
        self.P2 = P2;
        self.L = float(L);              ##Physical length
        self.X = np.zeros(N+2);
        self.Y = np.zeros(M+2);
        self.set_FULL(False); ##Toggle full bounce back conditions. Also initialize h=dx, and initialize X and Y..
        self.set_viscosity(eta);

       
        
    def set_FULL(self, bool):
        N, M, L = self.N, self.M, self.L;
        if(bool):
            print("full not implemented");
##            print "Using Full Bounce-Back";
##            self.full = True;
##            self.h = float(L)/(N+1);
##            self.Collision = self.Collision_Full;
##            self.Stream = self.Stream_Full;  
##            self.X = self.h*np.arange(N+2);
##            self.Y = self.h*np.arange(M+2);

        else:
            print("Using Half Bounce-Back");
            self.full = False;
            self.h = float(L)/(N); 
            self.X = self.h*(-0.5 + np.arange(N+2));
            self.Y = self.h*(-0.5 + np.arange(M+2));
         


    def set_viscosity(self, eta):       ##Update viscosity and choose timestep s accordingly
        self.eta = eta;
        self.s = (self.tau - 0.5)*(self.h**2)*(self.cs**2)/self.eta;

    def set_reynolds(self, Re, v, L):
        print(v*L/Re)
        self.set_viscosity(v*L/Re);
        print(self.eta);
        
    def boundary(self): 
        N, M = self.N, self.M;
               ##Bounce on top and bottom walls (i.e. no-slip)        
        for I in [2, 5, 6]:
            ex, ey = self.dct[I];
##            print self.dct[I], ex, ey
            J = self.opp[I];
            X = np.arange(-1, N+1)
##                print " "
##                print np.transpose(self.F[2].f);
##                print " "
##                print " "
            self.f[I][X, -1] = self.fs[J][X, -1];
            self.f[J][X, M] = self.fs[I][X, M];

                
##                dP_bot = self.f[I][i, -1] + self.fs[J][i, -1];
##                print dP_bot
                
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
        rhoL = self.P1/self.cs**2;
        rhoR = self.P2/self.cs**2;
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

        
        
        
        
        
        
        
        
        
        


    def plotFlow2(self, i):
        h = self.h;
        X = self.X;
        Y = self.Y;
##        print Y;
        U = (self.h/self.s)*self.getImage(self.VX[i])
        V = (self.h/self.s)*self.getImage(self.VY[i])
        plt.streamplot(X, Y, U, V, density=1.4);
        plt.ylim(0, h*self.M);
        plt.xlim(0, h*self.N);
        plt.imshow(self.getImage(1-(self.solid==1)),  extent=[0, self.L, 0, self.L*float(self.M)/self.N], alpha=0.5, cmap='gray', origin="lower", aspect='auto');
        plt.gca().set_aspect('equal');

##        plot_Circle(30, self.M/2 - 1, self.h, 15);


    def plotFlow(self):
        h = self.h;
        X = self.X;
        Y = self.Y;
        U = (self.h/self.s)*self.getImage(self.vx)
        V = (self.h/self.s)*self.getImage(self.vy)
        plt.streamplot(X, Y, U, V, density=1.4);
        plt.ylim(0, h*self.M);
        plt.xlim(0, h*self.N);

##        plot_Circle(30, self.M/2 - 1, self.h**2, 15);
        plt.imshow(self.getImage(1-(self.solid==1)),  extent=[0, self.L, 0, self.L*float(self.M)/self.N], alpha=0.5, cmap='gray', origin="lower", aspect='auto');
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

    def plotProfile(self):
        h = self.h;
        N, M, L = self.N, self.M, self.L;
        H = (M+1)*float(L)/(N+1);
        cp = self.h/self.s;
        Yvals = np.arange(-1, M+1);
        
        X, Y = self.X, self.Y;
        
        U = self.vx;
        plt.plot(Y, U[N/2, :][Yvals], label="vx profile");
        EXACT =  0.5*(self.cs**2)*(self.P2 - self.P1)*Y*(Y - H)/(self.eta) ;
##        EXACT = ( (self.h/self.s)*(self.P2 - self.P1)*(self.cs**2)/(2*self.eta) )*(self.L**2-4Y**2);
        plt.plot(Y, EXACT, label="exact sol'n");
        plt.title("t = {}".format(self.t));
        plt.xlabel("y");
        plt.ylabel("v_x");
        
        plt.legend();
        return np.sqrt(sum( (U[N/2, :][Yvals] - EXACT)**2 ));
        

    def heatmap(self, thing):
        h = self.h;
        X = self.X;
        Y = self.Y;
        plt.imshow(self.getImage(thing), origin='lower', interpolation='nearest', extent=[0, self.L, 0, self.L*float(self.M)/self.N]);
        plt.colorbar();
        
def plot_Circle(x0, y0, h, R):
    x0 = x0+1
    y0 = y0+1
    R = R-1
    X = h*(x0 + np.arange(-R, R+1));
    Y_p = h*(y0 + np.sqrt(R**2 - np.arange(-R, R+1)**2));
    Y_m = h*(y0 - np.sqrt(R**2 - np.arange(-R, R+1)**2));
    plt.plot(X, Y_p, 'r');
    plt.plot(X, Y_m, 'r');
    
def nyu(N, M, mask, r, T):
    
    for t in range(-T, T+1):
        x0 = N/4;
        y0 = M/2;
        print(x0, y0);
        for j in range(-r, r+1):
            mask[x0-r+t, y0+j] = 0;
            mask[x0+r+t, y0+j] = 0;
            mask[x0-j+t, y0+j] = 0;
        
        x0 = N/2;
        y0 = M/2;
        mask[x0-j+t, y0 + int(np.sqrt(r**2 - j**2))];
            
        for j in range(0, r-M/16+1):
            mask[x0+j+t, y0+j+M/16] = 0;
            mask[x0-j+t, y0+j+M/16] = 0;
        for j in range(-r, M/16+1):
            mask[x0+t, y0+j] = 0;
        
        x0 = 3*N/4;
        y0 = M/2;
        print(x0, y0);
        
        for j in range(-T, r+1):
            mask[x0-r+t, y0+j] = 0;
            mask[x0+r+t, y0+j] = 0;
            mask[x0-j+t, y0 - int(np.sqrt(r**2 - j**2))] = 0;
            mask[x0+j+t, y0 - int(np.sqrt(r**2 - j**2))] = 0;
            mask[x0-j+t, y0+1 - int(np.sqrt(r**2 - j**2))] = 0;
            mask[x0+j+t, y0+1 - int(np.sqrt(r**2 - j**2))] = 0;
    return mask;

def update_anim(i, gas):
    plt.clf();
    gas.plotFlow2(i);
    
    
def decimate(mask):
    N, M = np.shape(mask)
    N2, M2 = ([int(N/2), int(M/2)])
    mask2 = np.ones([int(N/2), int(M/2)])
    for i in range(N2):
        for j in range(M2):
            if sum(mask[2*i:2*i+1, 2*j:2*j+1]) == 0:
                mask2[i, j] = 0
#     plt.imshow(mask2)
#     plt.show()
    return mask2

def fatten(mask):
    N, M = np.shape(mask)
    mask2 = np.ones([N, M])*mask
    for i in range(N):
        for j in range(M):
            if mask[i, j]==1:
                mask2[i-1:i+1, j-1:j+1] = 1
                
    return mask2

def blockify(mask):
    N, M = np.shape(mask)
    N2, M2 = ([int(N/2), int(M/2)])
    for i in range(N2):
        for j in range(M2):
            if sum(mask[2*i:2*i+1, 2*j:2*j+1])<2:
                mask[2*i:2*i+1, 2*j:2*j+1] = 0
            else:
                mask[2*i:2*i+1, 2*j:2*j+1] = 1
#     plt.imshow(mask2)
#     plt.show()
    return mask






if __name__ == '__main__':
    number_of_frames = 200;
    ###### Define parameters of channel flow
    P0 = 1.2
    dP = 0.15
    P1 = P0 + dP
    P2 = P0 - dP
#     N = M = 8
#     N = = M = 126
    N = M = 254
#     N = 120
#     M = 64
#     N = M = 60
####    N = 5
####    M = 5
    L = 100.0
    eta = 5.0
    r = 16
    gas = poiseuille(N, M, P1, P2, L, eta)
##    gas.set_reynolds(100, 1.0, 8.0)
    
   

    h = gas.h
    s = gas.s
    print("s =", s)
    print("h =", h)

    
    rho = (1.0/gas.cs**2)*np.ones([N+2, M+2]);
    rho[-1, :] = (P1/gas.cs**2)*np.ones(M+2);
    rho[N, :] = (P2/gas.cs**2)*np.ones(M+2);
    vx = 1*np.zeros([N+2,M+2]);
    vy = np.zeros([N+2,M+2]);


#     mask = np.ones([N+2, M+2]);
#     mask = nyu(N, M, mask, 12, 3)
#     gas.defineSolid(mask);


    ###### Set object mask for a half-cylinder or cylinder
#     mask = np.ones([N+2, M+2]);
   


#     x0, y0 = 2*r, M/2 - 1;
#     for i in range(-r, r+1):
#         for j in range(-r, r+1):
# ##        for j in range(-r, 0):
#             if( i**2 + j**2 < r**2 ):
#                 mask[int(x0+i), int(y0+j)] = 0;

#     plot_Circle(x0, y0, 1.0, r);
    
#     r=r-1;
    
#     Xcyl = h*(x0 + np.arange(-r, r+1));
#     Ycyl_p = h*(y0 + np.sqrt(r**2 - np.arange(-r, r+1)**2));
#     Ycyl_p = h*y0*np.ones(np.size(Xcyl));
#     Ycyl_m = h*(y0 - np.sqrt(r**2 - np.arange(-r, r+1)**2));
    
#     mask = 1 - mask
#     gas.addSolid(mask);

########### Load a mask from a numpy array (in this case, DLA)
#     mask = np.load('dla_2d.npy')
#     mask = decimate(mask)
#     mask = fatten(mask)
#     mask = blockify(mask)
# #     mask = decimate(mask)

#     mask = np.zeros([N+2, M+2])
#     mask[5, 20:40] = 1
    
    
    
#     gas.addSolid(mask);
    plt.imshow(gas.getImage(gas.solid), origin='lower');
    plt.colorbar();
    
    plt.show();
    
    ###### Do iteration and plot
    err = [];
    for I in range(9):
        gas.updateEq(rho, vx, vy);
        gas.f[I] = gas.feq[I];
    count = 0;
    start = time.time();
    for i in range(number_of_frames):
#     while(gas.t < s*number_of_frames):
        if i==100:
            mask = np.load('dla_2d.npy')
    
            gas.addSolid(mask);
#         print(gas.rho)
#         gas.plotF()
#         plt.imshow(gas.vy)
#         plt.colorbar()
#         plt.show()
        gas.iterate(s);
        print(count);
        gas.VX.append(gas.vx);
        gas.VY.append(gas.vy);

        update_anim(count, gas);
        count += 1;
##        print count
        if(count%1==0):
            plt.subplot(211);
##            gas.plotProfile();
            gas.plotVorticity()
##            gas.heatmap(gas.vx**2 + gas.vy**2);
##            gas.heatmap(gas.rho);
#             plot_Circle(x0, y0, gas.h, r);

##        gas.plotFlow();
##        plt.pause(0.01);
########            plt.show();
##        plt.clf();
##            plt.subplot(212);
            plt.subplot(212);
            gas.plotFlow();

#             plt.plot(gas.vx[:, int(M/2)] + gas.vy[:, int(M/2)]**2, label="vx at center");
#             plt.plot(gas.vx[:, -1] + gas.vy[:, -1]**2, label="vx at bottom");
#             plt.plot(gas.vx[:, M]**2 + gas.vy[:, M]**2, label="v^2 at top");
#             plt.xlabel("x")
#             plt.ylabel("v_x");
#             print("v_wall = {}v_center".format(max(gas.vx[:, int(M/2)])/max(gas.vx[:, -1])))
#             plt.legend();

#             plot_Circle(x0, y0, gas.h, r);
          
    
            plt.pause(0.1);
#             plt.clf();
##            
            
                                                                       
                                                    
####        if(count%30==0):
####            gas.plotProfile();
####            plt.pause(0.01);
####            plt.clf();

##            print "rho left = ", gas.rho[-1, :];
##            print "rho right = ", gas.rho[N, :];
##            print "vx bottom = ", gas.vx[:, -1];
##            print "vx top = ", gas.vx[:, M];
##            print " " 
##            print "t = {}, error = {}, error - TOL = {}".format(gas.t, myerr, myerr - TOL);

    stop = time.time();
    print("the time taken to run was", stop-start);


    fig4=plt.figure();

    anim = animation.FuncAnimation(fig4, update_anim, number_of_frames, fargs=(gas, ));
    anim.save('movie.gif', writer='ImageMagickWriter',fps=20)

    ##    for count in range(np.size(self.obstacles)):
    ##        print "Obstacle {} Force Calculations".format(count+1);
    ##        print "Fx per unit time: ", gas.Fx[count]*s/gas.t;
    ##        print "Fy per unit time: ", gas.Fy[count]*s/gas.t;
    ##



