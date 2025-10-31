"""
Bravais Lattice Sphere Generator and Highest-Order Intersection Finder

This module generates 3D coordinates of spheres arranged in Bravais lattices
(SC, BCC, FCC, HCP) and computes the maximum number of sphere surfaces that
intersect at a single point.

Core approach for intersection finding:
- For each triplet of spheres, solve analytically for circle intersection points
- Collect all intersection points from all triplets
- Cluster points to identify unique locations
- Count sphere surface intersections at each unique point
- Return the maximum count
"""

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cdist


def generate_bravais_lattice(
    lattice_type: str,
    a: float = 1.0,
    c: float = None,
    supercell: tuple = (2, 2, 2),
    radius = 0.5,
) -> tuple:
    """
    Generate a set of sphere centers and radii for a given Bravais lattice.

    Parameters
    ----------
    lattice_type : str
        One of {"sc", "bcc", "fcc", "hcp"}
    a : float
        Lattice parameter (in Å or arbitrary units). Default is 1.0.
    c : float, optional
        Optional c-axis parameter for hcp. If None, defaults to ideal ratio
        c = sqrt(8/3) * a for hcp.
    supercell : tuple of int
        Tuple (nx, ny, nz) defining how many unit cells to replicate.
        Default is (2, 2, 2).
    radius : float or array-like
        Sphere radius. Can be a single uniform radius (float) or an array
        of radii for the basis atoms (will be repeated for supercell atoms).

    Returns
    -------
    centers : np.ndarray
        Shape (N, 3) array of sphere centers in Cartesian coordinates.
    radii : np.ndarray
        Shape (N,) array of sphere radii.

    Raises
    ------
    ValueError
        If lattice_type is not recognized or supercell contains non-positive values.
    """
    lattice_type = lattice_type.lower()
    if lattice_type not in {"sc", "bcc", "fcc", "hcp"}:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    nx, ny, nz = supercell
    if not all(n > 0 for n in (nx, ny, nz)):
        raise ValueError(f"Supercell dimensions must be positive, got {supercell}")

    # Define lattice vectors and basis for each lattice type
    if lattice_type == "sc":
        lattice_vectors = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a],
        ])
        basis = np.array([[0, 0, 0]])

    elif lattice_type == "bcc":
        lattice_vectors = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a],
        ])
        basis = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ])

    elif lattice_type == "fcc":
        lattice_vectors = np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a],
        ])
        basis = np.array([
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ])

    elif lattice_type == "hcp":
        if c is None:
            c = np.sqrt(8.0 / 3.0) * a
        lattice_vectors = np.array([
            [a, 0, 0],
            [a / 2.0, np.sqrt(3) * a / 2.0, 0],
            [0, 0, c],
        ])
        basis = np.array([
            [0, 0, 0],
            [2.0 / 3.0, 1.0 / 3.0, 0.5],
        ])

    # Generate all lattice points in the supercell
    centers_list = []
    radii_list = []

    # Normalize radius input
    if isinstance(radius, (int, float)):
        basis_radii = np.full(len(basis), float(radius))
    else:
        basis_radii = np.asarray(radius)
        if len(basis_radii) != len(basis):
            raise ValueError(
                f"Radius array length {len(basis_radii)} does not match "
                f"basis size {len(basis)}"
            )

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_index = np.array([i, j, k])
                cell_origin = cell_index @ lattice_vectors

                for basis_idx, basis_pos in enumerate(basis):
                    center = cell_origin + basis_pos @ lattice_vectors
                    centers_list.append(center)
                    radii_list.append(basis_radii[basis_idx])

    centers = np.array(centers_list)
    radii = np.array(radii_list)

    return centers, radii


def _compute_sphere_triplet_intersections_debug(
    centers: np.ndarray, radii: np.ndarray, i: int, j: int, k: int, tol: float = 1e-6
) -> np.ndarray:
    """Debug version that prints intermediate steps."""
    c_i, c_j, c_k = centers[i], centers[j], centers[k]
    r_i, r_j, r_k = radii[i], radii[j], radii[k]

    print(f"\n=== Triplet ({i}, {j}, {k}) ===")
    print(f"Centers: {c_i}, {c_j}, {c_k}")
    print(f"Radii: {r_i}, {r_j}, {r_k}")

    a_ij = 2.0 * (c_j - c_i)
    b_ij = np.dot(c_j, c_j) - np.dot(c_i, c_i) + r_i**2 - r_j**2

    a_ik = 2.0 * (c_k - c_i)
    b_ik = np.dot(c_k, c_k) - np.dot(c_i, c_i) + r_i**2 - r_k**2

    A_planes = np.vstack([a_ij, a_ik])
    b_planes = np.array([b_ij, b_ik])

    print(f"Plane equations A @ x = b:")
    print(f"A =\n{A_planes}")
    print(f"b = {b_planes}")

    try:
        u, s, vt = np.linalg.svd(A_planes, full_matrices=True)
        print(f"SVD singular values: {s}")

        if len(s) < 2 or s[-1] / s[0] > 1e-4:
            print("Planes are nearly parallel - skipping")
            return np.empty((0, 3))

        line_dir = vt[-1, :]
        line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-12)
        print(f"Line direction: {line_dir}")

        A_pinv = np.linalg.pinv(A_planes)
        x0 = A_pinv @ b_planes
        print(f"Point on line: {x0}")

        dc = x0 - c_i
        a_coeff = np.dot(line_dir, line_dir)
        b_coeff = 2.0 * np.dot(dc, line_dir)
        c_coeff = np.dot(dc, dc) - r_i**2

        print(f"Quadratic: {a_coeff} * t² + {b_coeff} * t + {c_coeff} = 0")

        discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff
        print(f"Discriminant: {discriminant}")

        if discriminant < -tol:
            print("No real solutions")
            return np.empty((0, 3))

        discriminant = max(0, discriminant)
        t_vals = [
            (-b_coeff + np.sqrt(discriminant)) / (2.0 * a_coeff),
            (-b_coeff - np.sqrt(discriminant)) / (2.0 * a_coeff),
        ]
        print(f"t values: {t_vals}")

        intersections = []
        verify_tol = max(1e-4, tol * 100)

        for idx, t in enumerate(t_vals):
            point = x0 + t * line_dir

            dist_to_ci = np.linalg.norm(point - c_i)
            dist_to_cj = np.linalg.norm(point - c_j)
            dist_to_ck = np.linalg.norm(point - c_k)

            error_i = abs(dist_to_ci - r_i)
            error_j = abs(dist_to_cj - r_j)
            error_k = abs(dist_to_ck - r_k)

            print(f"  t={t}: point={point}")
            print(f"    Dist to centers: [{dist_to_ci:.6f}, {dist_to_cj:.6f}, {dist_to_ck:.6f}]")
            print(f"    Errors: [{error_i:.6f}, {error_j:.6f}, {error_k:.6f}], verify_tol={verify_tol}")

            if error_i < verify_tol and error_j < verify_tol and error_k < verify_tol:
                intersections.append(point)
                print(f"    --> ACCEPTED")
            else:
                print(f"    --> REJECTED")

        return np.array(intersections) if intersections else np.empty((0, 3))

    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e}")
        return np.empty((0, 3))


def _compute_sphere_pairwise_intersections(
    centers: np.ndarray, radii: np.ndarray, i: int, j: int, tol: float = 1e-6
) -> np.ndarray:
    """
    Compute intersection points of two sphere surfaces.

    Given two spheres with centers c_i, c_j and radii r_i, r_j,
    we solve:
        |x - c_i| = r_i
        |x - c_j| = r_j

    The intersection (if it exists) is a circle in 3D space.
    We return multiple points on that circle for clustering.

    Parameters
    ----------
    centers : np.ndarray
        (N, 3) array of sphere centers
    radii : np.ndarray
        (N,) array of sphere radii
    i, j : int
        Indices of the two spheres
    tol : float
        Tolerance for detecting degenerate configurations

    Returns
    -------
    intersection_points : np.ndarray
        Shape (M, 3) where M is 0 or multiple points on the circle
    """
    c_i, c_j = centers[i], centers[j]
    r_i, r_j = radii[i], radii[j]

    # Vector between centers
    d = c_j - c_i
    dist = np.linalg.norm(d)

    # Check if spheres actually intersect
    if dist > r_i + r_j + tol or dist < abs(r_i - r_j) - tol or dist < tol:
        # Spheres don't intersect or are identical
        return np.empty((0, 3))

    # The intersection is a circle. Find its center and radius.
    # Using the formula from analytic geometry:
    # Distance from c_i to the plane of the circle
    a = (dist**2 + r_i**2 - r_j**2) / (2.0 * dist)

    # Radius of the intersection circle
    circle_radius_sq = r_i**2 - a**2
    if circle_radius_sq < -tol:
        return np.empty((0, 3))

    circle_radius = np.sqrt(max(0, circle_radius_sq))

    # Center of the intersection circle
    circle_center = c_i + a * (d / dist)

    # If circle radius is very small, return just the center point
    if circle_radius < tol:
        return np.array([circle_center])

    # Generate multiple points on the circle (for clustering)
    # Create orthonormal basis in the plane perpendicular to d
    d_normalized = d / dist

    # Find two orthogonal vectors to d
    if abs(d_normalized[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])

    v1 = np.cross(d_normalized, u)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(d_normalized, v1)
    v2 = v2 / np.linalg.norm(v2)

    # Generate points around the circle
    n_points = max(2, int(np.ceil(circle_radius * 10)))  # More points for larger circles
    points = []
    for theta_idx in range(n_points):
        theta = 2.0 * np.pi * theta_idx / n_points
        point = circle_center + circle_radius * (np.cos(theta) * v1 + np.sin(theta) * v2)
        points.append(point)

    return np.array(points) if points else np.empty((0, 3))


def _compute_sphere_triplet_intersections(
    centers: np.ndarray, radii: np.ndarray, i: int, j: int, k: int, tol: float = 1e-6
) -> np.ndarray:
    """
    Compute intersection points of three sphere surfaces using a robust analytical method.

    Given three spheres with centers c_i, c_j, c_k and radii r_i, r_j, r_k,
    we solve:
        |x - c_i| = r_i
        |x - c_j| = r_j
        |x - c_k| = r_k

    Approach:
    1. Expand all three sphere equations: |x - c_i|² = r_i²
    2. Subtract the first equation from the other two to get two linear constraints
    3. These define two planes whose intersection is a line
    4. Find the point on this line closest to satisfying all three sphere equations
    5. Solve a quadratic for the distance along the line

    Parameters
    ----------
    centers : np.ndarray
        (N, 3) array of sphere centers
    radii : np.ndarray
        (N,) array of sphere radii
    i, j, k : int
        Indices of the three spheres
    tol : float
        Tolerance for detecting degenerate configurations

    Returns
    -------
    intersection_points : np.ndarray
        Shape (M, 3) where M is 0, 1, or 2
    """
    c_i, c_j, c_k = centers[i], centers[j], centers[k]
    r_i, r_j, r_k = radii[i], radii[j], radii[k]

    # Build the linear system from the two plane equations
    # Plane 1: 2(c_j - c_i) · x = |c_j|² - |c_i|² + r_i² - r_j²
    # Plane 2: 2(c_k - c_i) · x = |c_k|² - |c_i|² + r_i² - r_k²

    a_ij = 2.0 * (c_j - c_i)
    b_ij = np.dot(c_j, c_j) - np.dot(c_i, c_i) + r_i**2 - r_j**2

    a_ik = 2.0 * (c_k - c_i)
    b_ik = np.dot(c_k, c_k) - np.dot(c_i, c_i) + r_i**2 - r_k**2

    # Stack into a matrix: A @ x = b
    A_planes = np.vstack([a_ij, a_ik])
    b_planes = np.array([b_ij, b_ik])

    # Check if planes are identical
    norm_ij = np.linalg.norm(a_ij)
    norm_ik = np.linalg.norm(a_ik)
    
    if norm_ij < 1e-10 or norm_ik < 1e-10:
        # Degenerate plane (all coefficients ~0)
        return np.empty((0, 3))
    
    # Check if planes are parallel (normals are parallel)
    cross_product = np.cross(a_ij, a_ik)
    if np.linalg.norm(cross_product) < 1e-10:
        # Planes are parallel - no line of intersection or infinite solutions
        return np.empty((0, 3))

    # Planes are not parallel; find the line of intersection
    line_dir = cross_product / np.linalg.norm(cross_product)

    # Find a point on the line satisfying both plane equations
    # Use least-squares with pseudo-inverse
    try:
        A_pinv = np.linalg.pinv(A_planes)
        x0 = A_pinv @ b_planes
    except np.linalg.LinAlgError:
        return np.empty((0, 3))

    # Now parameterize points on the line as: x(t) = x0 + t * line_dir
    # Substitute into the first sphere equation: |x(t) - c_i|² = r_i²
    # Expand: |x0 - c_i + t * line_dir|² = r_i²
    #         |x0 - c_i|² + 2t * (x0 - c_i) · line_dir + t² * |line_dir|² = r_i²

    dc = x0 - c_i
    a_coeff = np.dot(line_dir, line_dir)  # Should be ~1 since normalized
    b_coeff = 2.0 * np.dot(dc, line_dir)
    c_coeff = np.dot(dc, dc) - r_i**2

    discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff

    if discriminant < -tol:
        return np.empty((0, 3))

    discriminant = max(0, discriminant)
    t_vals = [
        (-b_coeff + np.sqrt(discriminant)) / (2.0 * a_coeff),
        (-b_coeff - np.sqrt(discriminant)) / (2.0 * a_coeff),
    ]

    intersections = []
    verify_tol = max(1e-3, tol * 100)

    for t in t_vals:
        point = x0 + t * line_dir

        # Verify the point lies on all three spheres
        dist_to_ci = np.linalg.norm(point - c_i)
        dist_to_cj = np.linalg.norm(point - c_j)
        dist_to_ck = np.linalg.norm(point - c_k)

        error_i = abs(dist_to_ci - r_i)
        error_j = abs(dist_to_cj - r_j)
        error_k = abs(dist_to_ck - r_k)

        if error_i < verify_tol and error_j < verify_tol and error_k < verify_tol:
            intersections.append(point)

    return np.array(intersections) if intersections else np.empty((0, 3))


def max_sphere_surface_intersection_optimized(
    centers: np.ndarray, 
    radii: np.ndarray, 
    tol: float = 1e-6,
    early_stop_threshold: int = None,
    verbose: bool = True,
) -> int:
    """
    Optimized computation of highest-order intersection with early stopping.

    Strategy:
    1. Quick pairwise check to establish baseline
    2. If k_max >= early_stop_threshold, return immediately
    3. Only proceed to expensive triplet computation if beneficial
    4. Use adaptive clustering tolerance based on k_max found so far

    Parameters
    ----------
    centers : np.ndarray
        Shape (N, 3) array of sphere centers
    radii : np.ndarray
        Shape (N,) array of sphere radii
    tol : float
        Numerical tolerance (default 1e-6)
    early_stop_threshold : int, optional
        Stop searching if k_max reaches this value. If None, search all.
        Useful values: 4, 6, 8 depending on expected max coordination.
    verbose : bool
        Print progress information

    Returns
    -------
    k_max : int
        Maximum number of sphere surfaces intersecting at a single point
    """
    n_spheres = len(centers)

    if n_spheres < 3:
        return 0 if n_spheres == 0 else (1 if n_spheres == 1 else 2)

    if early_stop_threshold is None:
        early_stop_threshold = float('inf')

    all_intersection_points = []

    if verbose:
        print(f"Optimized k_max search for {n_spheres} spheres")

    # ========================================================================
    # PHASE 1: Quick Pairwise Check (O(N²))
    # ========================================================================
    if verbose:
        print("Phase 1: Pairwise intersections (fast, establishes baseline)...")
    
    total_pairs = n_spheres * (n_spheres - 1) // 2
    found_pair_intersections = 0
    
    for i in range(n_spheres):
        for j in range(i + 1, n_spheres):
            points = _compute_sphere_pairwise_intersections(
                centers, radii, i, j, tol=tol
            )
            if len(points) > 0:
                all_intersection_points.append(points)
                found_pair_intersections += 1

    if verbose:
        print(f"  Found {found_pair_intersections} pairs with intersections")

    # Preliminary k_max from pairwise only
    if all_intersection_points:
        all_points = np.vstack(all_intersection_points)
        k_max_from_pairs = _count_max_intersection_order(all_points, centers, radii, tol)
    else:
        k_max_from_pairs = 0

    if verbose:
        print(f"  k_max from pairs only: {k_max_from_pairs}")

    # Early exit if we've found sufficient coordination
    if k_max_from_pairs >= early_stop_threshold:
        if verbose:
            print(f"  Early stop: k_max={k_max_from_pairs} >= threshold={early_stop_threshold}")
        return k_max_from_pairs

    # ========================================================================
    # PHASE 2: Triplet Intersections (O(N³)) - Only if needed
    # ========================================================================
    if verbose:
        print("Phase 2: Triplet intersections (expensive, searching for k≥3)...")

    total_triplets = n_spheres * (n_spheres - 1) * (n_spheres - 2) // 6
    found_triplet_intersections = 0
    k_max_so_far = k_max_from_pairs

    for i in range(n_spheres):
        for j in range(i + 1, n_spheres):
            for k in range(j + 1, n_spheres):
                points = _compute_sphere_triplet_intersections(
                    centers, radii, i, j, k, tol=tol
                )
                if len(points) > 0:
                    all_intersection_points.append(points)
                    found_triplet_intersections += 1

                    # Check early stopping after every ~10% of triplets
                    if found_triplet_intersections % max(1, total_triplets // 10) == 0:
                        # Quick count to see if we've exceeded threshold
                        temp_all = np.vstack(all_intersection_points)
                        k_max_so_far = _count_max_intersection_order(temp_all, centers, radii, tol)
                        if k_max_so_far >= early_stop_threshold:
                            if verbose:
                                print(f"  Early stop at {found_triplet_intersections} triplets: k_max={k_max_so_far}")
                            return k_max_so_far

    if verbose:
        print(f"  Found {found_triplet_intersections} triplets with intersections")

    # ========================================================================
    # PHASE 3: Final Count
    # ========================================================================
    if not all_intersection_points:
        if verbose:
            print("No intersections found")
        return 0

    all_points = np.vstack(all_intersection_points)
    k_max = _count_max_intersection_order(all_points, centers, radii, tol)

    if verbose:
        print(f"Final result: k_max = {k_max}")

    return k_max


def _count_max_intersection_order(
    points: np.ndarray, 
    centers: np.ndarray, 
    radii: np.ndarray, 
    tol: float = 1e-6
) -> int:
    """
    Count maximum sphere surface intersection at clustered points.

    This is extracted as a separate function to enable efficient early stopping.

    Parameters
    ----------
    points : np.ndarray
        Shape (M, 3) all intersection points (not yet clustered)
    centers : np.ndarray
        Shape (N, 3) sphere centers
    radii : np.ndarray
        Shape (N,) sphere radii
    tol : float
        Tolerance for clustering and surface detection

    Returns
    -------
    k_max : int
        Maximum intersection order found
    """
    n_spheres = len(centers)

    # Cluster points
    if len(points) > 1:
        cluster_tol = max(tol * 100, 1e-4)
        clusters = fclusterdata(
            points, t=cluster_tol, criterion="distance", method="complete"
        )
        unique_clusters = np.unique(clusters)
    else:
        unique_clusters = np.array([1])

    # Count intersections at each unique point
    k_max = 0
    count_tol = max(1e-3, tol * 1000)

    for cluster_id in unique_clusters:
        if len(points) > 1:
            mask = clusters == cluster_id
            cluster_points = points[mask]
            representative_point = cluster_points.mean(axis=0)
        else:
            representative_point = points[0]

        # Count intersections at this point
        intersection_count = 0
        for i in range(n_spheres):
            dist_to_center = np.linalg.norm(representative_point - centers[i])
            distance_to_surface = abs(dist_to_center - radii[i])

            if distance_to_surface < count_tol:
                intersection_count += 1

        k_max = max(k_max, intersection_count)

    return k_max


def max_sphere_surface_intersection(
    centers: np.ndarray, radii: np.ndarray, tol: float = 1e-6
) -> int:
    """
    Compute the highest-order intersection among all sphere surfaces.

    Strategy:
    1. For each triplet of spheres, solve analytically for intersection points
    2. Collect all intersection points
    3. Cluster points within tolerance to identify unique locations
    4. For each unique point, count how many spheres satisfy |‖x - c_i‖ - r_i| < tol
    5. Return the maximum count

    Parameters
    ----------
    centers : np.ndarray
        Shape (N, 3) array of sphere centers
    radii : np.ndarray
        Shape (N,) array of sphere radii
    tol : float
        Tolerance for point clustering and sphere surface detection.
        Default is 1e-6.

    Returns
    -------
    k_max : int
        Maximum number of sphere surfaces intersecting at a single point
    """
    n_spheres = len(centers)

    if n_spheres < 3:
        return 0 if n_spheres == 0 else (1 if n_spheres == 1 else 2)

    # Collect all intersection points from all sphere pairs and triplets
    all_intersection_points = []

    print(f"Computing intersections for {n_spheres} spheres...")
    
    # Phase 1: Pairwise intersections (finds k=2 cases)
    print("Phase 1: Computing pairwise intersections...")
    total_pairs = n_spheres * (n_spheres - 1) // 2
    pair_count = 0
    found_pair_intersections = 0
    
    for i in range(n_spheres):
        for j in range(i + 1, n_spheres):
            points = _compute_sphere_pairwise_intersections(
                centers, radii, i, j, tol=tol
            )
            if len(points) > 0:
                all_intersection_points.append(points)
                found_pair_intersections += 1
            
            pair_count += 1
            if pair_count % max(1, total_pairs // 5) == 0:
                print(f"  Processed {pair_count}/{total_pairs} pairs")
    
    print(f"Found {found_pair_intersections} pairs with intersections")
    
    # Phase 2: Triplet intersections (finds k≥3 cases)
    print("Phase 2: Computing triplet intersections...")
    total_triplets = n_spheres * (n_spheres - 1) * (n_spheres - 2) // 6

    triplet_count = 0
    found_triplet_intersections = 0
    for i in range(n_spheres):
        for j in range(i + 1, n_spheres):
            for k in range(j + 1, n_spheres):
                points = _compute_sphere_triplet_intersections(
                    centers, radii, i, j, k, tol=tol
                )
                if len(points) > 0:
                    all_intersection_points.append(points)
                    found_triplet_intersections += 1

                triplet_count += 1
                if triplet_count % max(1, total_triplets // 5) == 0:
                    print(f"  Processed {triplet_count}/{total_triplets} triplets")

    print(f"Found {found_triplet_intersections} triplets with intersections")

    if not all_intersection_points:
        print("No triple intersections found.")
        return 0

    all_points = np.vstack(all_intersection_points)
    print(f"Found {len(all_points)} total intersection points from triplets")

    # Cluster points to identify unique locations
    if len(all_points) > 1:
        # Use hierarchical clustering with a distance threshold
        # Scale tolerance adaptively based on problem size
        cluster_tol = max(tol * 100, 1e-4)
        clusters = fclusterdata(
            all_points, t=cluster_tol, criterion="distance", method="complete"
        )
        unique_clusters = np.unique(clusters)
        n_unique_points = len(unique_clusters)
    else:
        n_unique_points = 1
        unique_clusters = np.array([1])

    print(f"Clustered into {n_unique_points} unique point locations")

    # For each unique point, count how many sphere surfaces pass through it
    k_max = 0
    details = []
    
    # Adaptive tolerance for counting intersections
    count_tol = max(1e-3, tol * 1000)

    for cluster_id in unique_clusters:
        if len(all_points) > 1:
            mask = clusters == cluster_id
            cluster_points = all_points[mask]
            representative_point = cluster_points.mean(axis=0)
        else:
            representative_point = all_points[0]

        # Count intersections at this point
        intersection_count = 0
        intersecting_spheres = []
        for i in range(n_spheres):
            dist_to_center = np.linalg.norm(representative_point - centers[i])
            distance_to_surface = abs(dist_to_center - radii[i])

            if distance_to_surface < count_tol:
                intersection_count += 1
                intersecting_spheres.append(i)

        if intersection_count > k_max:
            k_max = intersection_count
            details = {
                "point": representative_point.copy(),
                "count": intersection_count,
                "spheres": intersecting_spheres,
            }

    print(f"Maximum intersection order: {k_max}")
    if details:
        print(f"  Location: {details['point']}")
        print(f"  Intersecting spheres: {details['spheres']}")

    return k_max


def max_sphere_surface_intersection_fast(
    centers: np.ndarray,
    radii: np.ndarray,
    tol: float = 1e-6,
    target_k_max: int = None,
    verbose: bool = True,
) -> int:
    """
    Ultra-fast k_max finder using aggressive heuristics for parameter sweeps.

    Strategy:
    - If all radii are small: likely k_max will be low, only do pairwise
    - If some radii are large: likely k_max will be high, skip straight to triplets
    - Return as soon as k_max >= target

    This is optimized for scanning wide parameter spaces where you know
    the approximate range of k_max values.

    Parameters
    ----------
    centers : np.ndarray
        Shape (N, 3) array of sphere centers
    radii : np.ndarray
        Shape (N,) array of sphere radii
    tol : float
        Numerical tolerance (default 1e-6)
    target_k_max : int, optional
        Stop searching if k_max reaches this value. Useful for parameter sweeps
        where you want "good enough" answers quickly (e.g., target_k_max=6)
    verbose : bool
        Print minimal progress information

    Returns
    -------
    k_max : int
        Maximum intersection order (may be approximate if target_k_max was set)
    """
    n_spheres = len(centers)

    if n_spheres < 3:
        return 0 if n_spheres == 0 else (1 if n_spheres == 1 else 2)

    if target_k_max is None:
        target_k_max = float('inf')

    # Heuristic: Estimate whether to expect high or low k_max
    avg_radius = np.mean(radii)
    max_radius = np.max(radii)
    
    # Rough estimate: nearest neighbor distance in SC lattice is 1.0
    # If avg_radius > 0.4, expect some triplets. If > 0.6, expect many triplets.
    expect_high_k = max_radius > 0.6

    if verbose:
        print(f"Fast k_max search: {n_spheres} spheres, avg_r={avg_radius:.3f}, max_r={max_radius:.3f}")
        if expect_high_k:
            print("  → Expecting high k (skipping pairwise, going straight to triplets)")
        else:
            print("  → Expecting low k (pairwise first, then triplets if needed)")

    all_intersection_points = []

    # ========================================================================
    # Strategy A: Small radii - Quick pairwise only approach
    # ========================================================================
    if not expect_high_k:
        if verbose:
            print("Phase 1: Quick pairwise scan...")

        for i in range(n_spheres):
            for j in range(i + 1, n_spheres):
                points = _compute_sphere_pairwise_intersections(
                    centers, radii, i, j, tol=tol
                )
                if len(points) > 0:
                    all_intersection_points.append(points)

        if all_intersection_points:
            all_points = np.vstack(all_intersection_points)
            k_max = _count_max_intersection_order(all_points, centers, radii, tol)
        else:
            k_max = 0

        if k_max >= target_k_max:
            if verbose:
                print(f"Result: k_max = {k_max} (reached target)")
            return k_max

        if verbose:
            print(f"  k_max from pairs: {k_max}, proceeding to triplets...")

    # ========================================================================
    # Strategy B: Large radii OR pairwise didn't find enough - Triplet scan
    # ========================================================================
    if verbose:
        print("Phase 2: Triplet scan...")

    triplet_count = 0
    total_triplets = n_spheres * (n_spheres - 1) * (n_spheres - 2) // 6

    for i in range(n_spheres):
        for j in range(i + 1, n_spheres):
            for k in range(j + 1, n_spheres):
                points = _compute_sphere_triplet_intersections(
                    centers, radii, i, j, k, tol=tol
                )
                if len(points) > 0:
                    all_intersection_points.append(points)

                triplet_count += 1

                # Check early stopping periodically (every 20%)
                if triplet_count % max(1, total_triplets // 5) == 0:
                    if all_intersection_points:
                        temp_all = np.vstack(all_intersection_points)
                        k_max = _count_max_intersection_order(temp_all, centers, radii, tol)
                        if k_max >= target_k_max:
                            if verbose:
                                print(f"  Early stop: k_max = {k_max} (target reached)")
                            return k_max

    if not all_intersection_points:
        if verbose:
            print("Result: k_max = 0")
        return 0

    all_points = np.vstack(all_intersection_points)
    k_max = _count_max_intersection_order(all_points, centers, radii, tol)

    if verbose:
        print(f"Result: k_max = {k_max}")

    return k_max


def max_sphere_surface_intersection_sweep(
    centers_list: list,
    radii_list: list,
    tol: float = 1e-6,
    show_progress: bool = True,
) -> list:
    """
    Batch processing for parameter sweeps - highly optimized for many similar configs.

    This function processes multiple (centers, radii) pairs with shared computation
    where possible. Useful for scanning parameter space efficiently.

    Parameters
    ----------
    centers_list : list of np.ndarray
        List of center arrays, each shape (N, 3)
    radii_list : list of np.ndarray
        List of radii arrays, each shape (N,)
    tol : float
        Numerical tolerance
    show_progress : bool
        Print progress bar

    Returns
    -------
    k_max_list : list of int
        k_max for each configuration
    """
    n_configs = len(centers_list)
    k_max_results = []

    for idx, (centers, radii) in enumerate(zip(centers_list, radii_list)):
        if show_progress:
            print(f"  [{idx+1}/{n_configs}] ", end="", flush=True)

        # Use fast method for sweep
        k_max = max_sphere_surface_intersection_fast(
            centers, radii, tol=tol, verbose=False
        )
        k_max_results.append(k_max)

        if show_progress:
            print(f"k_max={k_max}")

    return k_max_results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Bravais Lattice Sphere Intersection Finder")
    print("=" * 70)

    # Example 1: FCC lattice 2x2x2
    print("\n--- Example 1: FCC (2x2x2) ---")
    centers, radii = generate_bravais_lattice(
        "fcc", a=1.0, supercell=(2, 2, 2), radius=0.5
    )
    print(f"Generated {len(centers)} sphere centers")
    print(f"Sample centers:\n{centers[:4]}")
    k_max = max_sphere_surface_intersection(centers, radii)
    print(f"\nResult: k_max = {k_max}\n")

    # Example 2: BCC lattice 2x2x2
    print("\n--- Example 2: BCC (2x2x2) ---")
    centers, radii = generate_bravais_lattice(
        "bcc", a=1.0, supercell=(2, 2, 2), radius=0.5
    )
    print(f"Generated {len(centers)} sphere centers")
    k_max = max_sphere_surface_intersection(centers, radii)
    print(f"\nResult: k_max = {k_max}\n")

    # Example 3: Simple cubic 2x2x2
    print("\n--- Example 3: SC (2x2x2) ---")
    centers, radii = generate_bravais_lattice(
        "sc", a=1.0, supercell=(2, 2, 2), radius=0.5
    )
    print(f"Generated {len(centers)} sphere centers")
    k_max = max_sphere_surface_intersection(centers, radii)
    print(f"\nResult: k_max = {k_max}\n")

    # Example 4: HCP lattice 2x2x2
    print("\n--- Example 4: HCP (2x2x2) ---")
    centers, radii = generate_bravais_lattice(
        "hcp", a=1.0, supercell=(2, 2, 2), radius=0.5
    )
    print(f"Generated {len(centers)} sphere centers")
    k_max = max_sphere_surface_intersection(centers, radii)
    print(f"\nResult: k_max = {k_max}\n")

    # Example 5: Smaller lattice with larger supercell
    print("\n--- Example 5: SC (3x3x3) ---")
    centers, radii = generate_bravais_lattice(
        "sc", a=1.0, supercell=(3, 3, 3), radius=0.5
    )
    print(f"Generated {len(centers)} sphere centers")
    k_max = max_sphere_surface_intersection(centers, radii)
    print(f"\nResult: k_max = {k_max}\n")

    print("=" * 70)
