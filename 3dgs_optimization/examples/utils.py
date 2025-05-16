import random
import torch
import numpy as np
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from pathlib import Path


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_intrinsics_matrix(fx, fy, cx, cy):
    return torch.tensor(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )


def load_glb_file(file_path):
    """
    Load a GLB file using trimesh and extract points and colors.

    Args:
        file_path: Path to the GLB file

    Returns:
        tuple: (points, colors) as numpy arrays
    """
    import trimesh
    import numpy as np

    # Load the GLB file using trimesh
    try:
        loaded_obj = trimesh.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise

    # Initialize arrays for collecting points and colors
    all_points = []
    all_colors = []

    # Handle both single mesh and scene with multiple meshes
    if isinstance(loaded_obj, trimesh.Scene):
        # It's a scene, process each mesh
        print(
            f"[load_points3d] GLB loaded as a scene with {len(loaded_obj.geometry)} meshes"
        )

        for mesh_name, mesh in loaded_obj.geometry.items():
            # Skip non-mesh objects
            if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                continue

            # Get vertices for this mesh
            mesh_points = np.array(mesh.vertices)
            mesh_colors = np.ones((mesh_points.shape[0], 3)) * 0.5  # Default gray

            # Handle colors based on what's available - with extra safety checks
            if hasattr(mesh, "visual"):
                if (
                    hasattr(mesh.visual, "vertex_colors")
                    and mesh.visual.vertex_colors is not None
                ):
                    if len(mesh.visual.vertex_colors) == len(mesh_points):
                        mesh_colors = mesh.visual.vertex_colors[:, :3] / 255.0
                elif (
                    hasattr(mesh.visual, "face_colors")
                    and mesh.visual.face_colors is not None
                ):
                    # If only face colors exist, expand to vertices
                    face_colors = mesh.visual.face_colors[:, :3] / 255.0
                    mesh_colors = np.zeros((mesh_points.shape[0], 3))
                    if hasattr(mesh, "faces") and mesh.faces is not None:
                        for face_idx, face in enumerate(mesh.faces):
                            if face_idx < len(face_colors):
                                for vertex_idx in face:
                                    if vertex_idx < len(mesh_colors):
                                        mesh_colors[vertex_idx] = face_colors[face_idx]
                elif (
                    hasattr(mesh.visual, "material")
                    and mesh.visual.material is not None
                ):
                    # Try to get material color
                    if (
                        hasattr(mesh.visual.material, "baseColorFactor")
                        and mesh.visual.material.baseColorFactor is not None
                    ):
                        try:
                            base_color = np.array(mesh.visual.material.baseColorFactor)
                            if base_color.size >= 3:  # Make sure we have at least RGB
                                # Just take the RGB part
                                base_color = base_color[:3]
                                mesh_colors = np.tile(
                                    base_color, (mesh_points.shape[0], 1)
                                )
                        except (TypeError, IndexError) as e:
                            print(f"Warning: Could not use baseColorFactor: {e}")

            # Add to our collections
            all_points.append(mesh_points)
            all_colors.append(mesh_colors)
    else:
        # It's a single mesh
        if not hasattr(loaded_obj, "vertices") or len(loaded_obj.vertices) == 0:
            raise ValueError("Loaded mesh has no vertices")

        mesh_points = np.array(loaded_obj.vertices)
        mesh_colors = np.ones((mesh_points.shape[0], 3)) * 0.5  # Default gray

        # Extract colors with extra safety checks
        if hasattr(loaded_obj, "visual"):
            if (
                hasattr(loaded_obj.visual, "vertex_colors")
                and loaded_obj.visual.vertex_colors is not None
            ):
                if len(loaded_obj.visual.vertex_colors) == len(mesh_points):
                    mesh_colors = loaded_obj.visual.vertex_colors[:, :3] / 255.0
            elif (
                hasattr(loaded_obj.visual, "face_colors")
                and loaded_obj.visual.face_colors is not None
            ):
                face_colors = loaded_obj.visual.face_colors[:, :3] / 255.0
                mesh_colors = np.zeros((mesh_points.shape[0], 3))
                if hasattr(loaded_obj, "faces") and loaded_obj.faces is not None:
                    for face_idx, face in enumerate(loaded_obj.faces):
                        if face_idx < len(face_colors):
                            for vertex_idx in face:
                                if vertex_idx < len(mesh_colors):
                                    mesh_colors[vertex_idx] = face_colors[face_idx]
            elif (
                hasattr(loaded_obj.visual, "material")
                and loaded_obj.visual.material is not None
            ):
                if (
                    hasattr(loaded_obj.visual.material, "baseColorFactor")
                    and loaded_obj.visual.material.baseColorFactor is not None
                ):
                    try:
                        base_color = np.array(
                            loaded_obj.visual.material.baseColorFactor
                        )
                        if base_color.size >= 3:  # Make sure we have at least RGB
                            # Just take the RGB part
                            base_color = base_color[:3]
                            mesh_colors = np.tile(base_color, (mesh_points.shape[0], 1))
                    except (TypeError, IndexError) as e:
                        print(f"Warning: Could not use baseColorFactor: {e}")

        all_points.append(mesh_points)
        all_colors.append(mesh_colors)

    # Combine all points and colors
    if not all_points:
        raise ValueError("No valid meshes found in the GLB file")

    # Concatenate if we have multiple meshes
    if len(all_points) == 1:
        points = all_points[0]
        colors = all_colors[0]
    else:
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)

    print(f"[load_points3d] GLB input: {points.shape[0]} points with colors")

    return points, colors


def load_points3d(
    points3d_path: Path,
    surface_sampling=False,
    mesh_input=False,
    upper_num=None,
    normalize_scale=False,
):
    try:
        import open3d as o3d
        import trimesh
    except ImportError:
        raise ImportError("Please install 'open3d' to read the file.")

    try:
        if surface_sampling:
            # Load mesh and validate triangles
            mesh = o3d.io.read_triangle_mesh(str(points3d_path))
            if not mesh.has_triangles():
                raise ValueError("Input mesh lacks triangles for surface sampling.")

            # Sample points uniformly from the mesh surface
            n_vertices = len(mesh.vertices)
            print(
                f"[load_points3d] Mesh input: {n_vertices} vertices, start surface sampling..."
            )
            target_num = upper_num if upper_num else 8 * n_vertices
            sampled_pcd = mesh.sample_points_uniformly(number_of_points=2 * target_num)
            sampled_pcd = mesh.sample_points_poisson_disk(
                number_of_points=target_num, pcl=sampled_pcd
            )

            # Extract points and colors
            points = np.asarray(sampled_pcd.points)
            if sampled_pcd.has_colors():
                colors = np.asarray(sampled_pcd.colors)
            else:
                colors = np.random.rand(*points.shape)
            print(
                f"[load_points3d] Mesh surface sampling: {points.shape[0]}/{n_vertices} points"
            )
        elif mesh_input:
            if points3d_path.suffix == ".glb":
                scene_or_mesh = trimesh.load(points3d_path)
                # If this is a multi-node scene, dump() merges them into a single mesh with transforms applied
                if isinstance(scene_or_mesh, trimesh.Scene):
                    # Combine everything into one triangulated mesh
                    mesh_trimesh = scene_or_mesh.dump(concatenate=True)
                else:
                    mesh_trimesh = scene_or_mesh
                # Now mesh_trimesh.vertices and mesh_trimesh.faces contain the *transformed* geometry
                points = np.array(mesh_trimesh.vertices)  # (N, 3)
                faces = np.array(mesh_trimesh.faces)  # (M, 3)
                if hasattr(mesh_trimesh.visual, "vertex_colors") and len(
                    mesh_trimesh.visual.vertex_colors
                ) == len(points):
                    rgba_colors = np.array(mesh_trimesh.visual.vertex_colors)  # (N, 4)
                    # Typically in [0..255]. Convert to [0..1] if you want floating colors:
                    colors = rgba_colors[:, :3] / 255.0  # (N, 3) in RGB
                    print("Found per-vertex colors, shape:", colors.shape)
                else:
                    colors = np.random.rand(*points.shape)  # (N, 3)
            else:
                mesh = o3d.io.read_triangle_mesh(str(points3d_path))
                points = np.asarray(mesh.vertices)
                if mesh.has_vertex_colors():
                    colors = np.asarray(mesh.vertex_colors)
                else:
                    colors = np.random.rand(*points.shape)
            print(f"[load_points3d] Mesh input: {points.shape[0]} points")
        else:
            # Load point cloud directly
            pcd = o3d.io.read_point_cloud(str(points3d_path))
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
            else:
                colors = np.random.rand(*points.shape)

        # subsample points if upper_num is specified
        if upper_num is not None and points.shape[0] > upper_num:
            # Randomly sample points
            idx = np.random.choice(len(points), upper_num, replace=False)
            points = points[idx]
            colors = colors[idx]
            print(
                f"[load_points3d] Randomly sampled {upper_num}/{points.shape[0]} points"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load file: {e}")

    # Apply normalization if requested
    if normalize_scale:
        # Calculate bounding box
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)

        # Calculate scale factor
        bbox_size = bbox_max - bbox_min
        scale_factor = 1.0 / np.max(bbox_size)

        # Apply scale to points
        points = points * scale_factor

        # Recalculate bounding box after scaling
        bbox_min_scaled = np.min(points, axis=0)
        bbox_max_scaled = np.max(points, axis=0)

        # Center points at origin
        offset = -(bbox_min_scaled + bbox_max_scaled) / 2
        points = points + offset

        print(
            f"[load_points3d] Applied normalization with scale factor: {scale_factor:.6f}"
        )

    return {"points": points, "colors": colors}


def load_npz_first(path):
    with np.load(path) as d:
        return d[list(d.keys())[0]]
