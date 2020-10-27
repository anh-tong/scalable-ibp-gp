import numpy as np
import tensorflow as tf
from gpflow.kernels.multioutput import MultioutputKernel
from gpflow import Parameter, default_float, default_jitter
from gpflow.utilities import triangular
from gpflow import covariances
from gpflow.conditionals.multioutput.conditionals import separate_independent_conditional
from gpflow.conditionals import conditional

class MultiSVGP(tf.Module):

    def __init__(self,
                 kernel: MultioutputKernel,
                 num_latent_gps):
        super().__init__("MultiSVGP")

        self.kernel = kernel
        self.num_latent_gps = num_latent_gps
        self.inducing_variables = None


    def init_variational_params(self, num_inducing):
        q_mu = np.zeros((num_inducing, self.num_kernels, self.num_latent_gps))  # M x K x O
        self.q_mu = Parameter(q_mu, dtype=default_float())

        q_sqrt = []
        for _ in range(self.num_kernels):
            q_sqrt.append(
                [
                    np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
                ]
            )
        q_sqrt = np.array(q_sqrt)
        self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # K x O x M x M





    @property
    def num_kernels(self):
        return self.kernel.num_latent_gps




    def __call__(self, Xnew, full_cov=False, full_output_cov=False):
        q_mu = self.q_mu # M x K x O
        q_sqrt = self.q_sqrt # K x O x M x M

        Kuu = covariances.Kuu(self.inducing_variables, self.kernel, jitter=default_jitter())  # K x M x M
        Kuf = covariances.Kuf(self.inducing_variables, self.kernel, Xnew)  # K x M x N
        Knn = self.kernel.K(Xnew, full_output_cov=False)







if __name__ == "__main__":

    from gpflow.kernels.multioutput import SeparateIndependent
    from gpflow.kernels import RBF

    kernel = SeparateIndependent([RBF(), RBF()])
    msvgp = MultiSVGP(kernel, 3)
    msvgp.init_variational_params(10)
    print(msvgp.q_mu)
    print(msvgp.q_sqrt)