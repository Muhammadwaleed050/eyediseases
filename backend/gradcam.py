import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm

class GradCAM:
    """
    Generic Grad-CAM for torchvision models.
    target_layer: a layer/module in the model to hook (e.g. model.features[-1] for EfficientNet)
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        # register_full_backward_hook is best for newer PyTorch
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out  # shape: (B, C, H, W)

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # shape: (B, C, H, W)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        """
        input_tensor: (1,3,H,W) normalized
        returns: cam (H,W) in [0,1]
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)  # (1,num_classes)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        acts = self.activations[0].detach()  # (C,h,w)
        grads = self.gradients[0].detach()   # (C,h,w)

        # weights: global-average-pool gradients over spatial dims
        weights = grads.mean(dim=(1, 2))  # (C,)

        cam = (weights[:, None, None] * acts).sum(dim=0)  # (h,w)
        cam = torch.relu(cam)

        # normalize to [0,1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

def overlay_heatmap_on_image(original_rgb: Image.Image, cam_01: np.ndarray, alpha=0.45):
    """
    original_rgb: PIL RGB image (any size)
    cam_01: numpy (h,w) in [0,1]
    returns: PIL RGB heatmap overlay image
    """
    w, h = original_rgb.size
    cam_img = Image.fromarray((cam_01 * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)

    cam_np = np.array(cam_img) / 255.0  # (H,W)
    heatmap = cm.jet(cam_np)[:, :, :3]  # (H,W,3) float [0,1]

    orig = np.array(original_rgb).astype(np.float32) / 255.0
    overlay = (1 - alpha) * orig + alpha * heatmap
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
