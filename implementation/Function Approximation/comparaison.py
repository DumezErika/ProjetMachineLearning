import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
from jax import vmap
from jax.api import jit, grad

def mlp(params, inputs):
    for w, b in params:
        outputs = jnp.dot(inputs, w) + b
        inputs = jnp.tanh(outputs)
    return outputs
    
def nn_dynamics(state, time, params):
  state_and_time = jnp.hstack([state, jnp.array(time)])
  return mlp(params, state_and_time)

def resnet(params, inputs, depth):
    for i in range(depth):
        outputs = mlp(params, inputs) + inputs
    return outputs

def odenet(params, input):
  start_and_end_times = jnp.array([0.0, 1.0])
  init_state, final_state = odeint(nn_dynamics, input, start_and_end_times, params)
  return final_state

def odenet_loss(params, inputs, targets):
  preds = batched_odenet(params, inputs)
  return jnp.mean(jnp.sum((preds - targets)**2, axis=1))

def odenet_update(params, inputs, targets, step_size):
  grads = grad(odenet_loss)(params, inputs, targets)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m,n), scale * rng.randn(n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def resnet_squared_loss(params, inputs, targets, resnet_depth):
    preds = resnet(params, inputs, resnet_depth)
    return jnp.mean(jnp.sum((preds - targets)**2, axis=1))

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m,n), scale * rng.randn(n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def resnet_update(params, inputs, targets, step_size, resnet_depth):
    grads = grad(resnet_squared_loss)(params, inputs, targets, resnet_depth)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

def graphe(nom, fine_inputs, fine_outputs, inputs, noisy_targets, color1 = "red",  multiple = False, fine_outputs2 = None, nom2 = "", color2 = "green"):
  fig = plt.figure(figsize=(6, 4), dpi=150)
  ax = fig.gca()
  ax.scatter(inputs, noisy_targets, lw=0.5, color='indigo')
  ax.plot(fine_inputs, fine_outputs, lw=0.5, color=color1, label=nom)
  if multiple:
    ax.plot(fine_inputs, fine_outputs2, lw=0.5, color=color2, label=nom2)
  fine_targets = fine_inputs**3 + 0.1 * fine_inputs
  ax.plot(fine_inputs, fine_targets, lw=0.5, color='blue', label = "Target")
  ax.set_xlabel('input')
  ax.set_ylabel('output')
  plt.legend()
  plt.show()

def show_evol(nbr_iter, inputs, noisy_targets, fine_inputs, fine_outputs, nom, intervalle):
  if nbr_iter % intervalle == 0:
    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.gca()
    ax.scatter(inputs, noisy_targets, lw=0.5, color='indigo')
    ax.plot(fine_inputs, fine_outputs, lw=0.5, color='red', label=nom)
    fine_targets = fine_inputs**3 + 0.1 * fine_inputs
    ax.plot(fine_inputs, fine_targets, lw=0.5, color='blue', label = "Target")
    ax.set_xlabel('input')
    ax.set_ylabel('output')
    plt.legend()
    plt.show()

if __name__ == "__main__":
  batched_odenet = vmap(odenet, in_axes=(None, 0))
  npr.seed(1)

  inputs = jnp.reshape(jnp.linspace(-2.5, 2.5, 10), (10, 1))
  targets = inputs**3 + 0.1 * inputs
  noise =  npr.randn(10)
  noise = jnp.reshape(noise, (10, 1))
  noisy_targets = targets + noise
  fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 100), (100, 1))

  # On doit changer la dimension de l'input à 2 pour permettre des dynamique qui dépendent du temps
  odenet_layer_sizes = [2, 20, 1]
  resnet_depth = 3

  # Hyperparamètres des couches
  layer_sizes = [1, 20, 1]
  param_scale = 1
  step_size = 0.01
  train_iters = 1000

  # Initialise les poids et biais des couches et entraine le modèle
  resnet_params = init_random_params(param_scale, layer_sizes)
  for i in range(train_iters):
    resnet_params = resnet_update(resnet_params, inputs, noisy_targets, step_size, resnet_depth)
    show_evol(i, inputs, noisy_targets, fine_inputs, resnet(resnet_params, fine_inputs, resnet_depth), "ResNet", 20)

  param_scale = 0.1
  step_size = 0.005
  train_iters = 5000

  odenet_params = init_random_params(param_scale, odenet_layer_sizes)

  for i in range(train_iters):
    odenet_params = odenet_update(odenet_params, inputs, noisy_targets, step_size)
    # show_evol(i, inputs, noisy_targets, fine_inputs, batched_odenet(odenet_params, fine_inputs), "ODE-Net", 100)

  fine_outputs_odenet = batched_odenet(odenet_params, fine_inputs)
  fine_outputs_resnet = resnet(resnet_params, fine_inputs, resnet_depth)
  graphe("ResNet", fine_inputs, fine_outputs_resnet, inputs, noisy_targets)
  graphe("ODE-Net", fine_inputs, fine_outputs_odenet, inputs, noisy_targets, color1 = "green")
  graphe("ResNet", fine_inputs, fine_outputs_resnet, inputs, noisy_targets, multiple=True, fine_outputs2=fine_outputs_odenet, nom2 = "ODE-Net")