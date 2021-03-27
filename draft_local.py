import torch

class someFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, input):
        ctx.save_for_backward(weight)
        return weight + input
    @staticmethod
    def backward(ctx, output_grad):
        weight = ctx.saved_tensors
        return output_grad, output_grad


def main1():
    # a = torch.randn((3, 4, 5), dtype=torch.float, requires_grad=True)
    # c = torch.sum(a)
    # c.backward()
    # print(a.grad[1, 1, 1])
    # a.grad[1, 1, 1] = 0.01
    # print(a.grad[1, 1, 1])
    a = torch.randn((3, 4, 5))
    b = torch.randn((4, 5))
    print((a + b)[0, 0, 0])
    print(a[0, 0, 0] + b[0, 0])

if __name__ == '__main__':
    main1()