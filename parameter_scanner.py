"""
Parameter Scanner for Lattice Intersections

Provides systematic parameter space exploration and visualization capabilities
for crystal structure analysis. Supports multi-parameter sweeps, phase diagram
generation, and data persistence.
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Callable
from itertools import product
import time


@dataclass
class ScanPoint:
    """A single data point from a parameter scan."""
    params: Dict[str, float]
    k_max: int
    n_spheres: int
    computation_time: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ScanResult:
    """Results from a parameter scan operation."""
    scan_type: str
    parameters_varied: List[str]
    fixed_parameters: Dict[str, float]
    points: List[ScanPoint]
    total_time: float
    
    def to_dict(self):
        return {
            'scan_type': self.scan_type,
            'parameters_varied': self.parameters_varied,
            'fixed_parameters': self.fixed_parameters,
            'points': [p.to_dict() for p in self.points],
            'total_time': self.total_time,
        }
    
    def to_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved results to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        points = [ScanPoint(**p) for p in data['points']]
        return cls(
            scan_type=data['scan_type'],
            parameters_varied=data['parameters_varied'],
            fixed_parameters=data['fixed_parameters'],
            points=points,
            total_time=data['total_time'],
        )


class ParameterScanner:
    """
    Systematic parameter space scanner for lattice intersection analysis.
    
    Supports:
    - 1D sweeps (radius, lattice parameter, supercell size)
    - 2D phase diagrams (radius vs offset, radius vs lattice parameter, etc.)
    - N-dimensional parameter spaces
    - Progress reporting and early stopping
    - Result caching and persistence
    """
    
    def __init__(self, lattice_gen_func, k_max_func, verbose: bool = True):
        """
        Initialize the scanner.
        
        Parameters
        ----------
        lattice_gen_func : callable
            Function to generate lattice: lattice_gen_func(**params) -> (centers, radii)
        k_max_func : callable
            Function to compute k_max: k_max_func(centers, radii) -> int
        verbose : bool
            Print progress information
        """
        self.lattice_gen_func = lattice_gen_func
        self.k_max_func = k_max_func
        self.verbose = verbose
        self.scan_cache = {}
    
    def _generate_and_compute(self, params: Dict[str, float]) -> Tuple[int, int, float]:
        """
        Generate lattice for given parameters and compute k_max.
        
        Returns
        -------
        k_max : int
        n_spheres : int
        computation_time : float
        """
        start_time = time.time()
        
        try:
            centers, radii = self.lattice_gen_func(**params)
            n_spheres = len(centers)
            k_max = self.k_max_func(centers, radii)
        except Exception as e:
            if self.verbose:
                print(f"  Error at {params}: {e}")
            return None, 0, time.time() - start_time
        
        elapsed = time.time() - start_time
        return k_max, n_spheres, elapsed
    
    def scan_1d(
        self,
        param_name: str,
        param_range: Tuple[float, float],
        n_points: int,
        fixed_params: Dict[str, float],
        scale: str = 'linear',
        show_progress: bool = True,
    ) -> ScanResult:
        """
        Scan a single parameter.
        
        Parameters
        ----------
        param_name : str
            Name of parameter to vary (e.g., 'radius')
        param_range : tuple of (min, max)
            Range for parameter
        n_points : int
            Number of points to sample
        fixed_params : dict
            Fixed parameters for lattice generation
        scale : str
            'linear' or 'log' scaling
        show_progress : bool
            Print progress bar
        
        Returns
        -------
        ScanResult
        """
        start_time = time.time()
        
        if scale == 'linear':
            values = np.linspace(param_range[0], param_range[1], n_points)
        elif scale == 'log':
            values = np.logspace(
                np.log10(param_range[0]),
                np.log10(param_range[1]),
                n_points
            )
        else:
            raise ValueError(f"Unknown scale: {scale}")
        
        points = []
        
        if self.verbose and show_progress:
            print(f"\nScanning {param_name} ({scale} scale, {n_points} points)")
            print(f"  Range: [{param_range[0]:.4f}, {param_range[1]:.4f}]")
            print(f"  Fixed: {fixed_params}")
            print("-" * 70)
            print(f"{'Param':<15} {'k_max':<10} {'Spheres':<12} {'Time (s)':<12}")
            print("-" * 70)
        
        for idx, val in enumerate(values):
            params = {param_name: val, **fixed_params}
            k_max, n_spheres, elapsed = self._generate_and_compute(params)
            
            if k_max is not None:
                point = ScanPoint(params, k_max, n_spheres, elapsed)
                points.append(point)
                
                if self.verbose and show_progress:
                    print(f"{val:<15.6f} {k_max:<10} {n_spheres:<12} {elapsed:<12.4f}")
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print("-" * 70)
            print(f"Completed {len(points)} points in {total_time:.2f}s\n")
        
        return ScanResult(
            scan_type='1D',
            parameters_varied=[param_name],
            fixed_parameters=fixed_params,
            points=points,
            total_time=total_time,
        )
    
    def scan_2d(
        self,
        param_names: Tuple[str, str],
        param_ranges: Tuple[Tuple[float, float], Tuple[float, float]],
        n_points: Tuple[int, int],
        fixed_params: Dict[str, float],
        scales: Tuple[str, str] = ('linear', 'linear'),
        show_progress: bool = True,
    ) -> ScanResult:
        """
        Scan two parameters to generate a 2D phase diagram.
        
        Parameters
        ----------
        param_names : tuple of str
            Names of parameters to vary
        param_ranges : tuple of (min, max) tuples
            Ranges for each parameter
        n_points : tuple of int
            Number of points for each parameter
        fixed_params : dict
            Fixed parameters
        scales : tuple of str
            Scaling ('linear' or 'log') for each parameter
        show_progress : bool
            Print progress
        
        Returns
        -------
        ScanResult
        """
        start_time = time.time()
        
        # Generate parameter grids
        grids = []
        for param_range, n, scale in zip(param_ranges, n_points, scales):
            if scale == 'linear':
                grid = np.linspace(param_range[0], param_range[1], n)
            elif scale == 'log':
                grid = np.logspace(
                    np.log10(param_range[0]),
                    np.log10(param_range[1]),
                    n
                )
            else:
                raise ValueError(f"Unknown scale: {scale}")
            grids.append(grid)
        
        total_configs = n_points[0] * n_points[1]
        
        if self.verbose:
            print(f"\nScanning 2D phase space: {param_names[0]} × {param_names[1]}")
            print(f"  {param_names[0]}: {param_ranges[0]} ({n_points[0]} points, {scales[0]})")
            print(f"  {param_names[1]}: {param_ranges[1]} ({n_points[1]} points, {scales[1]})")
            print(f"  Fixed: {fixed_params}")
            print(f"  Total configurations: {total_configs}")
            print("-" * 70)
        
        points = []
        completed = 0
        
        for val1, val2 in product(grids[0], grids[1]):
            params = {
                param_names[0]: val1,
                param_names[1]: val2,
                **fixed_params
            }
            k_max, n_spheres, elapsed = self._generate_and_compute(params)
            
            if k_max is not None:
                point = ScanPoint(params, k_max, n_spheres, elapsed)
                points.append(point)
            
            completed += 1
            if self.verbose and show_progress and completed % max(1, total_configs // 10) == 0:
                pct = 100 * completed / total_configs
                print(f"  Progress: {completed}/{total_configs} ({pct:.0f}%)")
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print("-" * 70)
            print(f"Completed {len(points)} configurations in {total_time:.2f}s\n")
        
        return ScanResult(
            scan_type='2D',
            parameters_varied=list(param_names),
            fixed_parameters=fixed_params,
            points=points,
            total_time=total_time,
        )
    
    def scan_nd(
        self,
        param_specs: List[Tuple[str, Tuple[float, float], int, str]],
        fixed_params: Dict[str, float],
        show_progress: bool = True,
    ) -> ScanResult:
        """
        Scan N parameters (Cartesian product of all parameter ranges).
        
        Parameters
        ----------
        param_specs : list of (name, (min, max), n_points, scale)
            Specifications for each varying parameter
        fixed_params : dict
            Fixed parameters
        show_progress : bool
            Print progress
        
        Returns
        -------
        ScanResult
        """
        start_time = time.time()
        
        param_names = []
        grids = []
        
        for name, param_range, n, scale in param_specs:
            param_names.append(name)
            
            if scale == 'linear':
                grid = np.linspace(param_range[0], param_range[1], n)
            elif scale == 'log':
                grid = np.logspace(
                    np.log10(param_range[0]),
                    np.log10(param_range[1]),
                    n
                )
            else:
                raise ValueError(f"Unknown scale: {scale}")
            grids.append(grid)
        
        # Calculate total configurations
        total_configs = 1
        for n in [spec[2] for spec in param_specs]:
            total_configs *= n
        
        if self.verbose:
            print(f"\nScanning {len(param_names)}D parameter space")
            for name, param_range, n, scale in param_specs:
                print(f"  {name}: [{param_range[0]:.4f}, {param_range[1]:.4f}] ({n} points, {scale})")
            print(f"  Fixed: {fixed_params}")
            print(f"  Total configurations: {total_configs}")
            print("-" * 70)
        
        points = []
        completed = 0
        
        for values in product(*grids):
            params = dict(zip(param_names, values))
            params.update(fixed_params)
            
            k_max, n_spheres, elapsed = self._generate_and_compute(params)
            
            if k_max is not None:
                point = ScanPoint(params, k_max, n_spheres, elapsed)
                points.append(point)
            
            completed += 1
            if self.verbose and show_progress and completed % max(1, total_configs // 10) == 0:
                pct = 100 * completed / total_configs
                print(f"  Progress: {completed}/{total_configs} ({pct:.0f}%)")
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print("-" * 70)
            print(f"Completed {len(points)} configurations in {total_time:.2f}s\n")
        
        return ScanResult(
            scan_type=f'{len(param_names)}D',
            parameters_varied=param_names,
            fixed_parameters=fixed_params,
            points=points,
            total_time=total_time,
        )
    
    def adaptive_scan_1d(
        self,
        param_name: str,
        param_range: Tuple[float, float],
        initial_points: int = 5,
        refinement_threshold: float = 0.5,
        max_iterations: int = 5,
        fixed_params: Dict[str, float] = None,
    ) -> ScanResult:
        """
        Adaptively scan a parameter, refining regions of rapid change.
        
        Uses initial coarse sampling, then iteratively refines regions where
        k_max changes rapidly.
        
        Parameters
        ----------
        param_name : str
            Parameter to vary
        param_range : tuple of (min, max)
        initial_points : int
            Initial coarse sample points
        refinement_threshold : float
            Relative change threshold for refinement
        max_iterations : int
            Maximum refinement iterations
        fixed_params : dict
            Fixed parameters
        
        Returns
        -------
        ScanResult
        """
        if fixed_params is None:
            fixed_params = {}
        
        if self.verbose:
            print(f"\nAdaptive scan of {param_name}")
            print(f"  Range: [{param_range[0]:.4f}, {param_range[1]:.4f}]")
            print(f"  Initial points: {initial_points}")
            print(f"  Threshold: {refinement_threshold}")
        
        all_points = []
        current_values = np.linspace(param_range[0], param_range[1], initial_points)
        
        for iteration in range(max_iterations):
            if self.verbose:
                print(f"\n  Iteration {iteration + 1}: Testing {len(current_values)} points")
            
            iteration_points = []
            for val in current_values:
                if not any(np.isclose(p.params[param_name], val) for p in all_points):
                    params = {param_name: val, **fixed_params}
                    k_max, n_spheres, elapsed = self._generate_and_compute(params)
                    
                    if k_max is not None:
                        point = ScanPoint(params, k_max, n_spheres, elapsed)
                        iteration_points.append(point)
                        all_points.append(point)
            
            if iteration < max_iterations - 1 and len(iteration_points) > 1:
                # Find regions where k_max changes rapidly
                k_max_vals = np.array([p.k_max for p in iteration_points])
                refined_values = []
                
                for i in range(len(k_max_vals) - 1):
                    change = abs(k_max_vals[i+1] - k_max_vals[i]) / max(1, k_max_vals[i])
                    
                    if change > refinement_threshold:
                        # Refine between these points
                        v1 = iteration_points[i].params[param_name]
                        v2 = iteration_points[i+1].params[param_name]
                        refined_values.extend([
                            v1 + (v2 - v1) * 0.33,
                            v1 + (v2 - v1) * 0.67,
                        ])
                
                if refined_values:
                    current_values = np.array(refined_values)
                else:
                    if self.verbose:
                        print(f"  No regions need refinement")
                    break
            else:
                break
        
        # Sort by parameter value
        all_points.sort(key=lambda p: p.params[param_name])
        
        if self.verbose:
            print(f"  Total points: {len(all_points)}")
        
        return ScanResult(
            scan_type='1D-Adaptive',
            parameters_varied=[param_name],
            fixed_parameters=fixed_params,
            points=all_points,
            total_time=sum(p.computation_time for p in all_points),
        )


def export_to_csv(result: ScanResult, filepath: str):
    """Export scan results to CSV."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        # Collect all parameter names
        param_names = sorted(result.points[0].params.keys()) if result.points else []
        fieldnames = param_names + ['k_max', 'n_spheres', 'computation_time']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for point in result.points:
            row = {**point.params, 'k_max': point.k_max, 'n_spheres': point.n_spheres, 'computation_time': point.computation_time}
            writer.writerow(row)
    
    print(f"Exported to CSV: {filepath}")


def export_phase_diagram_data(result: ScanResult, filepath: str):
    """Export 2D scan as simple format suitable for matplotlib."""
    if result.scan_type != '2D':
        raise ValueError(f"Expected 2D scan, got {result.scan_type}")
    
    param_names = result.parameters_varied
    
    with open(filepath, 'w') as f:
        f.write(f"# 2D Phase Diagram: {param_names[0]} vs {param_names[1]}\n")
        f.write(f"# {param_names[0]}\t{param_names[1]}\tk_max\tn_spheres\n")
        
        for point in result.points:
            p1 = point.params[param_names[0]]
            p2 = point.params[param_names[1]]
            f.write(f"{p1:.6f}\t{p2:.6f}\t{point.k_max}\t{point.n_spheres}\n")
    
    print(f"Exported phase diagram data: {filepath}")
