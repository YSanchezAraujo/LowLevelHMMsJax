import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Tuple, List
from jax import jit, lax

def log_normalize(log_prob: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalizes log probabilities
    
    Args:
        log_prob: A vector of log probabilities
    
    Returns:
        A tuple containing the normalized log probabilities and the log of the normalization constant.
    """
    log_c = logsumexp(log_prob)
    return log_prob - log_c, log_c


@jit
def log_forward_message(
    log_lik_obs: jnp.ndarray,
    log_pi0: jnp.ndarray,
    log_A: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    """
    Computes the forward messages for a Hidden Markov Model.
    
    Args:
        log_lik_obs: Log likelihoods of observations, shape (n_steps, n_states).
        log_pi0: Log initial state probabilities, shape (n_states,).
        log_A: Log transition matrix, shape (n_states, n_states).
    
    Returns:
        An instance of LogForward containing log_alpha and log_c.
    """
    n_steps, _ = log_lik_obs.shape

    def scan_step(carry, step):
        prev_log_alpha,  = carry
        log_alpha_step, log_c_step = log_normalize(
            log_lik_obs[step, :] + logsumexp(log_A + prev_log_alpha[:, jnp.newaxis], axis=0)
            )
        return (log_alpha_step, ), (log_alpha_step, log_c_step)

    initial_log_alpha, initial_log_c = log_normalize(log_lik_obs[0, :] + log_pi0)
    initial_carry = (initial_log_alpha, )

    _, scan_output = lax.scan(scan_step, initial_carry, jnp.arange(1, n_steps))

    log_alpha, log_c = scan_output
    log_alpha = jnp.vstack([initial_log_alpha, log_alpha])
    log_c = jnp.hstack([initial_log_c, log_c])

    return log_alpha, log_c

@jit
def log_backward_message(
    log_lik_obs: jnp.ndarray, 
    log_A: jnp.ndarray, 
    log_c: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the backward messages for a Hidden Markov Model.
    
    Args:
        log_lik_obs: Log likelihoods of observations, shape (n_steps, n_states).
        log_A: Log transition matrix, shape (n_states, n_states).
        log_c: Log normalization constants from forward messages, shape (n_steps,).
    
    Returns:
        An instance of LogBackward containing log_beta.
    """
    n_steps, n_states = log_lik_obs.shape

    def scan_step(carry, step):
        prev_log_beta, = carry
        log_beta_sum = prev_log_beta + log_lik_obs[step+1, :]
        log_beta_step = logsumexp(log_A.T + log_beta_sum[:, jnp.newaxis], axis=0) - log_c[step+1]
        return (log_beta_step, ), log_beta_step

    initial_log_beta = jnp.zeros(n_states) - log_c[-1]
    initial_carry = (initial_log_beta, )
    _, scan_output = lax.scan(scan_step, initial_carry, jnp.arange(n_steps-2, -1, -1))
    log_beta = jnp.vstack([jnp.flip(scan_output, axis=0), initial_log_beta])

    return log_beta


@jit
def expectations(
    log_alpha: jnp.ndarray,
    log_beta: jnp.ndarray,
    log_c: jnp.ndarray,
    log_lik_obs: jnp.ndarray, 
    log_A: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the expectations (xi and gamma) for the Hidden Markov Model.
    
    Args:
        log_alpha: Instance of LogForward containing forward messages.
        log_beta: Instance of LogBackward containing backward messages.
        log_c: Log normalization constants from forward messages, shape (n_steps,).
        log_lik_obs: Log likelihoods of observations, shape (n_steps, n_states).
        log_A: Log transition matrix, shape (n_states, n_states).
    
    Returns:
        A dictionary containing:
            - 'xi': Expected transitions, shape (n_states, n_states).
            - 'gamma': Expected states, shape (n_steps, n_states).
            - 'pi0': Initial state probabilities, shape (n_states,).
    """
    n_steps, _ = log_lik_obs.shape

    log_gamma = log_alpha + log_beta 
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)

    def compute_xi(step):
        log_b_lik = log_lik_obs[step + 1, :] + log_beta[step + 1, :] 
        log_xi_step = log_alpha[step, :] + log_b_lik[:, jnp.newaxis] + log_A - log_c[step + 1]
        return jnp.exp(log_xi_step)

    xi = jax.vmap(compute_xi)(jnp.arange(n_steps - 1))
    xi = jnp.sum(xi, axis=0) 
    pi0 = gamma[0, :] 

    return xi.T, gamma, pi0
