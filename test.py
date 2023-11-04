import torch, torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections.abc import Iterable

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# make deterministic
# from PA2.sequenceutils import set_seed
from sequenceutils import set_seed
set_seed(42)

# imports for attention model
# more imports
# from PA2.attentionmodel import GPT, GPTConfig
# from PA2.sequenceutils import sample, CharDataset
# from PA2.attentiontrainer import Trainer, TrainerConfig
from attentiontrainer import Trainer, TrainerConfig
from attentionmodel import GPT, GPTConfig
from sequenceutils import sample, CharDataset

# Basic SGD implementation for reference.
# for base class, see https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
# for official pytorch SGD implementation, see https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py

class SGD(Optimizer):
  def __init__(self, params, lr=1.0):
    super(SGD, self).__init__(params, {'lr': lr})

    # The params argument can be a list of pytorch variables, or
    # a list of dicts. If it is a list of dicts, each dict should have 
    # a key 'params' that is a list of pytorch variables,
    # and optionally another key 'lr' that specifies the learning rate
    # for those variables. If 'lr' is not provided, the default value
    # is the single value provided as an argument after params to this
    # constructor.
    # If params is just a list of pytorch variables, it is the same
    # as if params were actually a list containing a single dictionary
    # whose 'params' key value is the list of variables.
    # See examples in following code blocks for use of params.

    # Set up an iteration counter.
    # self.state[p] is a python dict for each parameter p
    # that can be used to store various state useful in the optimization
    # algorithm. In this case, we simply store the iteration count, although
    # it is not used in this simple algorithm.
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'] = 0


  @torch.no_grad()
  def step(self, closure=None):
    '''
    closure is a function that computes the loss and returns it.
    '''
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr = group['lr']
      # it is common practice to call the model parameters p in code.
      # in class we follow more closely analytical conventions, in which the
      # parameters are often called w for weights.
      for p in group['params']:
          # p: params:,[w,b,..]
        if p.grad is None:
          continue
        
        # Update the iteration counter (again, this is not actually used in this algorithm)
        state = self.state[p]
        step = state['step']
        step += 1
        
        # Perform the SGD update. p.grad holds the gradient of the loss
        # with respect to p.
        p -= lr * p.grad
    
    return 

#Simple linear regression problem
dimension = 10
num_iter = 10000

mean = torch.zeros(dimension)
std = torch.ones(dimension)

def loss_func(w_hat, b_hat, w_true, b_true):
  # simple linear regression problem, although
  # slightly non-standard loss. See pytorch docs
  # for description of loss function.

  # features
  x = torch.normal(mean, std) #generate a guassian vector with mean and std 

  # true label is a linear function of features plus noise.
  noise = np.random.normal(0.0, 0.01)
  y_true = torch.dot(x, w_true) + b_true + noise

  y_hat = torch.dot(x, w_hat) + b_hat

  loss = torch.nn.SmoothL1Loss()
  return loss(y_hat, y_true)


# Set "true" parameter value to be a random normal vector with covariance 10*I
w_true = 10*torch.normal(mean, std)

# make true bias term quite large so that it is better
# to have a high learning rate for the bias. This makes
# it advantageous to use the params as a dict in the 
# following cell.
b_true = torch.normal(torch.zeros(1), torch.ones(1))


# declare variables that will actually be trained.
# "requires_grad" tells pytorch that it may have to compute
# gradients with respect to these variables so that it initialize the 
# relevant autograd data structures.
w = torch.zeros(dimension, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = SGD([w, b], 0.010)

losses = []

for t in range(num_iter):

  # "zero_grad" resets all variable.grad values to 0 and resets all
  # intermediate data that might have been saved for a backwards pass.
  # This is useful when variables need to be reused for many backward passes.
  # Note that your PA1 autograd implementation did not need to have this
  # functionality because the SGD you implemented just created  new
  # variables every iteration.
  optimizer.zero_grad()

  # Compute the loss function - this is the forward pass
  loss = loss_func(w, b, w_true, b_true)

  # loss.backward has essentially the same functionality as the .backward
  # function you implemented in PA1.
  loss.backward()

  optimizer.step()

  losses.append(loss.item())

# plt.plot(losses)
# plt.ylabel('loss')
# plt.xlabel('iteration')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()