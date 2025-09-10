import torch


def get_points_3d(bev, m_per_pixle, upsampling_steps, z_min=0, z_max=0, z_points=1):
    """
    Generates 3D points from the BEV grid.
    Args:
        bev (torch.Tensor): A tensor of shape (C, H, W) representing the BEV grid.
        m_per_pixle (float): The meters per pixel in the BEV grid.
        h_min (float): The minimum height in meters.
        h_max (float): The maximum height in meters.
        z_points (int): The number of points in the z-direction.
    Returns:
        torch.Tensor: A tensor of shape (H, W, Z, 3) representing the 3D points.
    """
    _, _, H, W = bev.shape
    x_start = (((H - 1) / 2) * m_per_pixle) * 2**upsampling_steps
    y_start = (
        ((W - 1) / 2) * m_per_pixle
    ) * 2**upsampling_steps  # (((W - 1) / 2) * m_per_pixle) * 14 * 2**upsampling_steps

    x = torch.linspace(x_start, -x_start, H, device=bev.device)
    y = torch.linspace(y_start, -y_start, W, device=bev.device)
    z = torch.linspace(z_min, z_max, z_points, device=bev.device)

    # Create 3D grid using meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    # Stack to get the combined (x, y, z) points with shape (H, W, Z, 3)
    points_3d = torch.stack([X, Y, Z], dim=-1)

    return points_3d


def get_value_tokens(
    points_3d, fmaps_g, extrinsics, intrinsics, resolutions, deformable_offsets=None
):  # Implement deformable_offsets in transformer implementation
    """
    B : BEV grid (C, H, W) where (H, W) are the BEV grid dimensions
    fmaps_g : Tensor of PV feature maps for each camera (B, N, C, H_fmap, W_fmap)
    extrinsics : Tensor of extrinsic camera parameters (B, N, 4, 4)
    intrinsics : Tensor of intrinsic camera parameters (B, N, 3, 3)
    resolutions : Tensor of camera resolutions (B, N, 2)
    stride : Stride of the PV feature maps
    deformable_offsets : Learnable offsets for deformable sampling (B, H, W, Z, 2)
    """
    H, W = points_3d.shape[:2]
    value_tokens_batched = []
    value_tokens_mask_batched = []

    # Iterate through batch
    for b, (fmap_b, extr_b, intr_b, res_b) in enumerate(
        zip(fmaps_g, extrinsics, intrinsics, resolutions)
    ):
        uv, valid_mask = project_3d_to_2d(points_3d, extr_b, intr_b, res_b)

        grid, pseudo_tokens_count = uv_to_grid(
            uv, valid_mask, res_b, fmap_b.shape[3]
        )  # Shape of grid (1, H, W*T, 2) with T beeing the number of tokens

        fmaps_concat = concat_fmaps(fmap_b)

        value_tokens = torch.nn.functional.grid_sample(
            fmaps_concat, grid, align_corners=True
        )  # Shape (1, C, H, W*T)
        value_tokens = (
            value_tokens.squeeze(0)
            .permute(1, 2, 0)
            .view(H, W, -1, value_tokens.shape[1])
        )  # Shape (H, W, T, C) , with T beeing the number of tokens
        value_tokens_mask = get_output_mask(pseudo_tokens_count)

        value_tokens_batched.append(value_tokens)
        value_tokens_mask_batched.append(value_tokens_mask)

    value_tokens = torch.stack(value_tokens_batched, dim=0)
    value_tokens_mask = torch.stack(value_tokens_mask_batched, dim=0)

    return (
        value_tokens,
        value_tokens_mask,
    )


def project_3d_to_2d(points_3d, extr, intr, res, offset=None):
    """
    Projects 3D points to 2D image coordinates using given extrinsic and intrinsic
    camera parameters.
    Args:
        points_3d (torch.Tensor): A tensor of shape (H, W, D, 3) representing the 3D
        points.
        extr (torch.Tensor): A tensor of shape (N, 4, 4) representing the extrinsic
        parameters for N cameras.
        intr (torch.Tensor): A tensor of shape (N, 3, 3) representing the intrinsic
        parameters for N cameras.
        res (torch.Tensor): A tensor of shape (N, 2) representing the resolution
        (height, width) for N cameras.
    Returns:
        tuple:
            uv (torch.Tensor): A tensor of shape (N, H, W, D, 2) representing the 2D
            image coordinates for each camera.
            valid_mask (torch.Tensor): A tensor of shape (N, H, W, D) indicating valid
            points that are within the image bounds and have positive z-values.
    """
    cam_count = extr.shape[0]
    H, W, D, _ = points_3d.shape

    # Reshape and augment points_3d for homogeneous coordinates
    points_3d_h = points_3d.view(-1, 3)  # Shape (H*W*D, 3)
    points_3d_h = torch.cat(
        [
            points_3d_h,
            torch.ones(points_3d_h.shape[0], device=points_3d_h.device).unsqueeze(1),
        ],
        dim=1,
    )  # Shape (H*W*D, 4)

    # Add dimension for N cameras
    points_3d_h = points_3d_h.unsqueeze(0).expand(
        cam_count, -1, -1
    )  # Shape (N, H*W*D, 4)

    # Transform points to camera coordinates
    points_3d_cam = torch.matmul(points_3d_h, extr.transpose(1, 2))[
        :, :, :3
    ]  # Shape (N, H*W*D, 3)

    # Get mask for points with positive z
    valid_mask_z = points_3d_cam[:, :, 2] > 0  # Shape (N, H*W*D)

    # Apply intrinsic transformation
    pixel_coords_h = torch.matmul(
        points_3d_cam, intr.transpose(1, 2)
    )  # Shape (N, H*W*D, 3)

    # Normalize by z
    uv = pixel_coords_h[:, :, :2] / pixel_coords_h[:, :, 2].unsqueeze(
        2
    )  # Shape (N, H*W*D, 2)

    # Check if u and v are within the image bounds
    img_height, img_width = (
        res[:, 0].unsqueeze(1),
        res[:, 1].unsqueeze(1),
    )  # Assuming resolution = (height, width)
    valid_mask_uv = (
        (uv >= 0).all(dim=2) & (uv[:, :, 0] < img_width) & (uv[:, :, 1] < img_height)
    )  # Shape (N, H*W*D)

    # Combine valid masks
    valid_mask = valid_mask_z & valid_mask_uv  # Shape (N, H*W*D)

    # Reshape uv and valid_mask to match shape of points_3d
    uv = uv.view(cam_count, H, W, D, 2)  # Shape (N, H, W, D, 2)
    valid_mask = valid_mask.view(cam_count, H, W, D)  # Shape (N, H, W, D)

    return uv, valid_mask


def uv_to_grid(uv, valid_mask, resolution, fmap_width):
    """
    Transforms UV coordinates to a grid format suitable for further processing.
    Args:
        uv (torch.Tensor): A tensor of shape (N, H, W, D, 2) representing UV
        coordinates.
        valid_mask (torch.Tensor): A boolean tensor of shape (N, H, W, D) indicating
        valid UV points.
        resolution (torch.Tensor): A tensor of shape (2,) representing the resolution.
        fmap_width (int): The width of the feature map.
    Returns:
        torch.Tensor: The transformed UV grid of shape (1, H, W*D, 2).
        torch.Tensor: The count of pseudo tokens for each (H, W) position.
        torch.Tensor: The padded UV coordinates of shape
        (H, W, N*D+max_valid_points, 2).
        torch.Tensor: The padded valid mask of shape (H, W, N*D+max_valid_points).
    """

    N, H, W, D, _ = uv.shape

    # In-place division to save memory
    n, _ = resolution.shape
    res_view = resolution.flip(-1).view(n, 1, 1, 1, 2)
    uv.div_(res_view)  # In-place division
    uv.mul_(fmap_width)  # In-place multiplication

    # In-place addition for offsets, using broadcast to avoid memory duplication
    offsets = torch.arange(N, device=uv.device).unsqueeze(1).unsqueeze(2).unsqueeze(
        3
    ) * (fmap_width + 1)
    uv[..., 0].add_(offsets)  # Modify in-place to avoid allocating new memory

    # Normalize uv coordinates to [-1, 1] range in-place
    uv[..., 0].div_(N * (fmap_width + 1)).mul_(2).sub_(
        1
    )  # In-place normalization for u
    uv[..., 1].div_(fmap_width).mul_(2).sub_(1)  # In-place normalization for v

    # Reshape UV and valid_mask in a memory-efficient way
    uv = uv.permute(1, 2, 0, 3, 4).contiguous()  # Ensure contiguous after permute
    valid_mask = valid_mask.permute(1, 2, 0, 3).contiguous()

    # In-place reshape to H, W, N*D, 2 and H, W, N*D
    uv = uv.view(H, W, N * D, -1)
    valid_mask = valid_mask.view(H, W, N * D)

    # Compute the max number of valid points
    valid_points_sum = valid_mask.sum(dim=2)
    max_valid_points = valid_points_sum.max().item()
    pseudo_tokens_count = max_valid_points - valid_points_sum

    # Create pseudo mask with minimal memory footprint using bool type
    pseudo_mask = torch.zeros(
        (H, W, max_valid_points), dtype=torch.bool, device=uv.device
    )
    indices = torch.arange(
        max_valid_points, device=uv.device, dtype=torch.int16
    ).expand(H, W, max_valid_points)
    pseudo_mask.copy_(indices < pseudo_tokens_count.unsqueeze(-1))  # In-place update

    padded_uv = torch.empty((H, W, N * D + max_valid_points, 2), device=uv.device)
    padded_uv[..., : N * D, :].copy_(uv)  # In-place copy of uv
    padded_uv[..., N * D :, :].fill_(-2)  # Fill the rest with -2 in-place

    # Concatenate valid masks
    padded_mask = torch.cat([valid_mask, pseudo_mask], dim=-1)

    # Mask the padded UV coordinates and reshape to the required format
    masked_uv = padded_uv[padded_mask].view(H, W, -1, 2)
    grid = masked_uv.view(H, -1, 2).unsqueeze(0)  # Shape (1, H, W*D, 2)

    return grid, pseudo_tokens_count  # padded_uv and padded_mask for demo purposes


def concat_fmaps(fmaps) -> torch.Tensor:
    """
    Concatenates the feature maps for each camera into a single tensor.
    Args:
        fmaps (torch.Tensor): A tensor of shape (N, C, H, W) representing the feature
        maps for each camera.
    Returns:
        torch.Tensor: A tensor of shape (1, C, H, N*(W+1)) representing the
        concatenated feature maps.
    """
    N, C, H, W = fmaps.shape

    # Create a view of zeros along the width axis directly in memory-efficient manner
    fmaps_padded = torch.nn.functional.pad(
        fmaps, (0, 1)
    )  # Pads last dimension (width) by 1 with zeros

    # Perform concatenation and reshaping in a single step avoid intermediate operations
    return fmaps_padded.permute(1, 2, 0, 3).contiguous().view(1, C, H, N * (W + 1))


def get_output_mask(pseudo_tokens_count) -> torch.Tensor:
    """
    Generates an output mask based on the pseudo to               kens count.
    Args:
        pseudo_tokens_count (torch.Tensor): A 2D tensor of shape (H, W) containing the
        count of pseudo tokens.
    Returns:
        torch.Tensor: A boolean mask tensor of shape (H, W, max_pseudo_tokens) where
        each element indicates whether the corresponding pseudo token index is less than
        the difference between the maximum pseudo tokens and the pseudo tokens count at
        that position.
    """
    H, W = pseudo_tokens_count.shape
    max_pseudo_tokens = pseudo_tokens_count.max().item()

    # Create a mask by comparing indices directly with adjusted pseudo tokens count
    mask = torch.arange(max_pseudo_tokens, device=pseudo_tokens_count.device).unsqueeze(
        0
    ).unsqueeze(0) < (max_pseudo_tokens - pseudo_tokens_count.unsqueeze(-1))

    return mask.expand(H, W, -1)  # Expand to (H, W, max_pseudo_tokens)
