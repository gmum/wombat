import torch


def get_ig_gradient_paths(model, x, edge_index, edge_features, n_steps: int = 100):
    all_grads_x = []
    all_grads_e = []
    preds = []
    all_activations = []

    for i in range(1, n_steps + 1):
        scale = i / n_steps

        x_modified = x * scale
        edge_features_modified = edge_features * scale

        x_modified = x_modified.clone()
        edge_features_modified = edge_features_modified.clone()

        x_modified.requires_grad = True
        edge_features_modified.requires_grad = True

        out, activations = model(x_modified, edge_features_modified, edge_index, dump_activations=True)
        out.backward()

        gradients_x = x_modified.grad
        gradients_edge_features = edge_features_modified.grad

        gradients_x = gradients_x * (1 / n_steps)
        gradients_edge_features = gradients_edge_features * (1 / n_steps)

        # Baselines are all zeros, so
        gradients_x = gradients_x * x
        gradients_edge_features = gradients_edge_features * edge_features

        all_grads_x.append(gradients_x)
        all_grads_e.append(gradients_edge_features)
        preds.append(out.item())
        all_activations.append(activations)

    all_grads_x = torch.stack(all_grads_x, dim=2)
    all_grads_e = torch.stack(all_grads_e, dim=2)

    return all_grads_x, all_grads_e, preds, all_activations
