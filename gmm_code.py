'''
Code Adapted from https://github.com/plgreenLIRU/semi_supervised_learning_tutorial
'''
class GMM:

    def __init__(self, X, mu_init, C_init, pi_init, N_mixtures):
        """ Initialiser class method
        """

        self.X = np.vstack(X)   # Inputs always vertically stacked
        self.mu = mu_init       # Initial means of Gaussian mixture
        self.C = C_init         # Initial covariance matrices
        self.pi = pi_init       # Initial mixture proportions
        self.N_mixtures = N_mixtures      # No. components in mixture
        self.N, self.D = np.shape(self.X)  # No. data points and dimension of X
        self.EZ = np.zeros([self.N, N_mixtures])  # Initialise expected labels

    def expectation(self):
        """ The 'E' part of the EM algorithm.
        Finds the expected labels of each data point.
        """

        for n in range(self.N):
            den = 0.0
            for k in range(self.N_mixtures):
                den += self.pi[k] * multivariate_normal.pdf(self.X[n],
                                                            self.mu[k],
                                                            self.C[k])
            for k in range(self.N_mixtures):
                num = self.pi[k] * multivariate_normal.pdf(self.X[n],
                                                           self.mu[k],
                                                           self.C[k])
                self.EZ[n, k] = num/den
                
        #print(self.EZ.shape)

    def maximisation(self, X, L):
        """ The 'M' part of the EM algorithm.
        Finds the maximum likelihood parameters of our model.
        Here we use 'L' to represent labels.
        """

        for k in range(self.N_mixtures):
            Nk = np.sum(L[:, k])
            self.pi[k] = Nk / self.N

            # Note - should vectorise this next bit in the future as
            # it will be a lot faster
            self.mu[k] = 0.0
            for n in range(self.N):
                self.mu[k] += 1/Nk * L[n, k]*X[n]
            self.C[k] = np.zeros([self.D, self.D])
            for n in range(self.N):
                self.C[k] += 1/Nk * L[n, k] * (np.vstack(X[n] - self.mu[k])
                                               * (X[n]-self.mu[k]))

    def train(self, Ni):
        """ Train Gaussian mixture model using the EM algorithm.
        """

        print('Training...')
        for i in range(Ni):
            print('Iteration', i)
            self.expectation()
            self.maximisation(self.X, self.EZ)

    def plot(self):
        """ Method for plotting results.
        Only 2D for now. Points only coloured in for problems
        with 2 mixtures.
        """

        if self.D == 2:

            # Plot contours
            r1 = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 1]), 100)
            r2 = np.linspace(np.min(self.X[:, 1]), np.max(self.X[:, 1]), 100)
            x_r1, x_r2 = np.meshgrid(r1, r2)
            pos = np.empty(x_r1.shape + (2, ))
            pos[:, :, 0] = x_r1
            pos[:, :, 1] = x_r2
            for k in range(self.N_mixtures):
                p = multivariate_normal(self.mu[k], self.C[k])
                plt.contour(x_r1, x_r2, p.pdf(pos))

            # Plot data
            if (self.N_mixtures == 2):
                for i in range(self.N):
                    plt.plot(self.X[i, 0], self.X[i, 1], 'o',
                             markerfacecolor=[0, self.EZ[i, 0],
                                              1 - self.EZ[i, 0]],
                             markeredgecolor='black')
            else:
                plt.plot(self.X[:, 0], self.X[:, 1], 'o',
                         markerfacecolor='red',
                         markeredgecolor='black')

        else:
            print('Currently only produce plots for 2D problems.')

            
            
class GMM_SemiSupervised(GMM):

    def __init__(self, X, X_labelled, Y, mu_init, C_init, pi_init,
                 N_mixtures):
        """ Initialiser class method
        """


        self.X = np.vstack(X)
        self.X_labelled = np.vstack(X_labelled)
        self.X_all = np.vstack((self.X_labelled, self.X))
        self.Y = np.vstack(Y)
        self.mu = mu_init
        self.C = C_init
        self.pi = pi_init
        self.N_mixtures = N_mixtures
        self.N_labelled, self.D = np.shape(self.X_labelled)
        self.N = np.shape(X)[0]
        self.EZ = np.zeros([self.N, N_mixtures]) # Initialise expected labels

    def train(self, Ni):
        """ Train (using EM)
        """

        print('Training...')
        for i in range(Ni):
            print('Iteration', i)
            self.expectation()
            L = np.vstack((self.Y, self.EZ))
            self.maximisation(self.X_all, L)

    def plot(self):
        """ Plots (just for 2D where no. of mixtures is 2 for now)
        """

        super().plot()
        if ((self.D == 2) and (self.N_mixtures == 2)):
            for i in range(self.N_labelled):
                if self.Y[i, 0] == 1:
                    plt.plot(self.X_labelled[i, 0], self.X_labelled[i, 1], 'v',
                             markerfacecolor=[0, 1, 0],
                             markeredgecolor='black',
                             markersize=10)
                else:
                    plt.plot(self.X_labelled[i, 0], self.X_labelled[i, 1], 'v',
                             markerfacecolor=[0, 0, 1],
                             markeredgecolor='black',
                             markersize=10)
    def predict(self,x_test):
        N_samples = x_test.shape[0]
        EZ_test = np.zeros([N_samples,self.N_mixtures])
        for n in range(N_samples):
            den = 0.0
            for k in range(self.N_mixtures):
                den += self.pi[k] * multivariate_normal.pdf(x_test[n,:],self.mu[k],self.C[k])
            for k in range(self.N_mixtures):
                num = self.pi[k] * multivariate_normal.pdf(x_test[n,:],self.mu[k],self.C[k])
                EZ_test[n, k] = num/den
        return np.argmax(EZ_test,axis=1)