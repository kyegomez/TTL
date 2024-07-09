import torch
from torch import nn, Tensor
from typing import List


class Task(nn.Module):
    """
    A class representing a task in a neural network model.

    Args:
        d1 (int): The dimension of the input data.
        d2 (int): The dimension of the output data.

    Attributes:
        d1 (int): The dimension of the input data.
        d2 (int): The dimension of the output data.
        t_k (nn.Parameter): The learnable parameter for key transformation.
        t_v (nn.Parameter): The learnable parameter for value transformation.
        t_q (nn.Parameter): The learnable parameter for query transformation.

    Methods:
        loss(f, x): Computes the loss between the predicted output and the target output.

    """

    def __init__(self, d1: int, d2: int):
        super(Task, self).__init__()
        self.d1 = d1
        self.d2 = d2

        # Theta
        self.t_k = nn.Parameter(torch.randn(1, d1, d2))
        self.t_v = nn.Parameter(torch.randn(1, d1, d2))
        self.t_q = nn.Parameter(torch.randn(1, d1, d2))

    def loss(self, f, x: Tensor) -> Tensor:
        """
        Computes the loss between the predicted output and the target output.

        Args:
            f (function): The function to apply to the transformed input.
            x (Tensor): The input data.

        Returns:
            Tensor: The computed loss.

        """
        train_view = self.t_k @ x
        label_view = self.t_v @ x
        return nn.functional.mse_loss(f(train_view), label_view)


class OGD(nn.Module):
    """
    Online Gradient Descent (OGD) optimizer.

    Args:
        lr (float): Learning rate for the optimizer.

    Attributes:
        lr (float): Learning rate for the optimizer.

    Methods:
        step(model, grad_in): Performs a single optimization step.

    """

    def __init__(self, lr: float = 0.01):
        super(OGD, self).__init__()
        self.lr = lr

    def step(self, model: nn.Module, grads: List[Tensor]):
        """
        Performs a single optimization step.

        Args:
            model (nn.Module): The model to optimize.
            grad_in (Tensor): The gradient of the loss with respect to the model parameters.

        """
        with torch.no_grad():
            for param, grad in zip(model.parameters(), grads):
                param -= self.lr * grad


class Learner(nn.Module):
    """
    Learner class represents a neural network model for a specific task.

    Args:
        task (Task): The task associated with the learner.
        dim (int): The input dimension of the model.
        output_dim (int): The output dimension of the model.

    Attributes:
        task (Task): The task associated with the learner.
        model (nn.Linear): The neural network model.
        optim (OGD): The optimizer for training the model.
    """

    def __init__(self, task: Task, dim: int, output_dim: int):
        super(Learner, self).__init__()
        self.task = task
        self.model = nn.Linear(dim, output_dim)
        self.optim = OGD()

    def train(self, x: Tensor):
        """
        Trains the learner model using the given input.

        Args:
            x (Tensor): The input tensor for training.

        Returns:
            None
        """
        loss = self.task.loss(self.model, x)

        grad_fn = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=True
        )

        self.optim.step(self.model, grad_fn)

    def predict(self, x: Tensor) -> Tensor:
        """
        Predicts the output for the given input.

        Args:
            x (Tensor): The input tensor for prediction.

        Returns:
            Tensor: The predicted output tensor.
        """
        view = self.task.t_q @ x
        return self.model(view)


class TTLinear(nn.Module):
    """
    TTLinear is a module that performs linear transformations on input sequences using Tensor Train (TT) decomposition.

    Args:
        d1 (int): The first dimension of the TT decomposition.
        d2 (int): The second dimension of the TT decomposition.
        input_dim (int): The dimension of the input sequence.
        output_dim (int): The dimension of the output sequence.

    Attributes:
        d1 (int): The first dimension of the TT decomposition.
        d2 (int): The second dimension of the TT decomposition.
        input_dim (int): The dimension of the input sequence.
        output_dim (int): The dimension of the output sequence.
        task (Task): The task object used for training and prediction.
        learner (Learner): The learner object used for training and prediction.

    Methods:
        forward(in_seq: List[Tensor]) -> List[Tensor]: Performs forward pass on the input sequence.

    """

    def __init__(self, input_dim: int, output_dim: int):
        super(TTLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = Task(input_dim, output_dim)

        self.learner = Learner(self.task, input_dim, output_dim)

    def forward(self, in_seq: List[Tensor]) -> List[Tensor]:
        """
        Performs forward pass on the input sequence.

        Args:
            in_seq (List[Tensor]): The input sequence.

        Returns:
            List[Tensor]: The output sequence.

        """
        out_seq = []

        # for each token in the input sequence
        # train the model and predict the output
        for tok in in_seq:
            self.learner.train(tok)
            out_seq.append(self.learner.predict(tok))
        return out_seq
