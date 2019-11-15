import torch
import torch.nn.functional as F


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer):
        """
        Args:
            model: ResNet_linear()
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def forward(self, x):
        """
        Args:
            x: input image. shape => (N, 3, T, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # object classification
        score = self.model(x)
        prob = F.softmax(score, dim=1)
        max_prob, idx = torch.max(prob, dim=1)
        print(
            "predicted action ids {}\t probability {}".format(idx.item(), max_prob.item()))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data
        cam = self.getCAM(self.values, weight_fc, idx.item())

        return cam

    def __call__(self, x):
        return self.forward(x)

    def getCAM(self, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (N, C, T, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        cam = F.conv3d(
            values.activations, weight=weight_fc[:, :, None, None, None])
        _, _, t, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, t, h, w)

        return cam.data


class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, T, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        score, _, _ = self.model(x)

        prob = torch.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAM(self.values, score, idx)

        return cam, idx

    def __call__(self, x, idx=None):
        return self.forward(x, idx)

    def getGradCAM(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, T, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, T, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data
