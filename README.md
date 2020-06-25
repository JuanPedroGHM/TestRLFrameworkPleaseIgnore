# Notes/Papers

[toc]

## Deep rl based finite-horizon optimal tracking control for nonlinear system

[Link](https://doi.org/10.1016/j.ifacol.2018.11.115)

**Cost/Value function**
$$
V(x_k, k) = \phi(x_N) + \sum^{N-1}_{i=k} \rho(x_i, u_i, i) \\
\rho(x_k, u_k, k) = ||h(x_k) - r_k||^2_Q + ||\Delta u_k||^2_R
$$

"Delta-input formulation can eliminate the offset computing the steady state input"????

**Costate funtion**
$$
\begin{array} \\
	\lambda (x_k, k) &= \frac{\partial V(x_k, k)}{\partial x_k} \\
	&= \min \left( \frac{\partial \rho(x_k, u_k, k)}{\partial x_k} + \left(\frac{\partial x_{k+1}}{\partial x_k}\right)^T \lambda^*(x_{k+1}, k+1) \right)
\end{array}
$$

After setting $\partial V^*(x_k, k)/\partial u_k = 0$:
$$
\Delta u_k^* = -\frac{1}{2} R^{-1}g^T(x_k)\lambda^*(x_{k+1}, k+1)
$$
Hamilton-Jacobi-Bellman PDE.



**Stable learning**

* Regularization: Minibatch + Dropout

* Replay buffer

* Target network

  * Update rule:
    $$
    \theta_t = \theta_c \tau + (1 - \tau) \theta_t
    $$

* 

## Deep deterministic Policy gradient

http://proceedings.mlr.press/v32/silver14.pdf

Q-Function value approximator biased is avoided if:

1. $$
   Q_\theta (s, a) = \nabla_\phi \log \pi_\phi (a | s) ^ T \theta
   $$

2. Parameters $\theta$  are chosen to minimize the MSE between the real Q-Function and $Q_\theta$ 



### Action-value gradients

$$
\nabla_\phi J (\pi_\phi) = E[\nabla_\phi \pi_\phi(s) \nabla_aQ_\theta(s, a) | a = \pi(s)]
$$

Deep version https://arxiv.org/pdf/1509.02971.pdf

## Comparison of algorithms

https://arxiv.org/pdf/1604.06778.pdf

## Checkout

[Multi Pseudo Q-learning Based Deterministic Policy Gradient for Tracking Control of Autonomous Underwater Vehicles](https://arxiv.org/pdf/1909.03204.pdf)

[RL for robotics](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Publications/IROS2006-Peters_[0].pdf)

[HIGH-DIMENSIONALCONTINUOUSCONTROLUSINGGENERALIZEDADVANTAGEESTIMATION](https://arxiv.org/pdf/1506.02438.pdf)

