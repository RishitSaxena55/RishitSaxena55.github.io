"""
Block-Diagonal Matrix Data Structure
=====================================
Technical Challenge Solution for Human Analysis Lab (HAL)
Michigan State University - Professor Vishnu Boddeti

Author: Rishit Saxena
Email: rishitsaxena55@gmail.com
Date: January 2026

Problem Statement:
------------------
A matrix A of size R^{nd × nd} where each non-overlapping d×d block D_{ij}
is a diagonal matrix. The matrix consists of n² such blocks:

    A = | D_11  D_12  ...  D_1n |
        | D_21  D_22  ...  D_2n |
        | ...   ...   ...  ...  |
        | D_n1  D_n2  ...  D_nn |

Key Insight:
------------
Since each D_ij is diagonal, we only need to store d values per block
(the diagonal elements), not d² values. This gives us n² × d storage
instead of (nd)² = n²d² storage.

Storage: O(n²d) instead of O(n²d²) — d times more efficient!

Time Complexity:
- Multiplication: O(n³d) instead of O(n³d³) — d² times faster!
- Inverse: O(n³d) instead of O(n³d³) — d² times faster!
"""

import numpy as np
from typing import Optional, Tuple


class BlockDiagonalMatrix:
    """
    Efficient data structure for matrices with diagonal block structure.
    
    A matrix A ∈ R^{nd × nd} where each d×d block is diagonal.
    
    Attributes:
        n (int): Number of blocks per row/column
        d (int): Size of each diagonal block
        blocks (ndarray): Shape (n, n, d) storing diagonal elements of each block
    """
    
    def __init__(self, n: int, d: int, blocks: Optional[np.ndarray] = None):
        """
        Initialize a BlockDiagonalMatrix.
        
        Args:
            n: Number of blocks per row/column
            d: Size of each diagonal block
            blocks: Optional array of shape (n, n, d) containing diagonal elements.
                   If None, initializes to zeros.
        """
        self.n = n
        self.d = d
        
        if blocks is not None:
            assert blocks.shape == (n, n, d), f"Expected shape {(n, n, d)}, got {blocks.shape}"
            self.blocks = blocks.copy()
        else:
            self.blocks = np.zeros((n, n, d))
    
    @classmethod
    def from_full_matrix(cls, matrix: np.ndarray, n: int, d: int) -> 'BlockDiagonalMatrix':
        """
        Create a BlockDiagonalMatrix from a full (nd × nd) matrix.
        
        Args:
            matrix: Full matrix of shape (nd, nd)
            n: Number of blocks per row/column
            d: Block size
            
        Returns:
            BlockDiagonalMatrix instance
        """
        assert matrix.shape == (n * d, n * d), f"Expected shape {(n*d, n*d)}, got {matrix.shape}"
        
        blocks = np.zeros((n, n, d))
        for i in range(n):
            for j in range(n):
                # Extract the (i,j)-th block
                block = matrix[i*d:(i+1)*d, j*d:(j+1)*d]
                # Store only diagonal elements
                blocks[i, j, :] = np.diag(block)
        
        return cls(n, d, blocks)
    
    @classmethod
    def random(cls, n: int, d: int, seed: Optional[int] = None) -> 'BlockDiagonalMatrix':
        """Create a random BlockDiagonalMatrix."""
        if seed is not None:
            np.random.seed(seed)
        blocks = np.random.randn(n, n, d)
        return cls(n, d, blocks)
    
    @classmethod
    def identity(cls, n: int, d: int) -> 'BlockDiagonalMatrix':
        """Create an identity BlockDiagonalMatrix (I_nd with block structure)."""
        blocks = np.zeros((n, n, d))
        for i in range(n):
            blocks[i, i, :] = 1.0
        return cls(n, d, blocks)
    
    def to_full_matrix(self) -> np.ndarray:
        """
        Convert to full (nd × nd) matrix for verification.
        
        Returns:
            Full matrix of shape (nd, nd)
        """
        full = np.zeros((self.n * self.d, self.n * self.d))
        for i in range(self.n):
            for j in range(self.n):
                # Place diagonal elements in the (i,j)-th block
                for k in range(self.d):
                    full[i*self.d + k, j*self.d + k] = self.blocks[i, j, k]
        return full
    
    def __repr__(self) -> str:
        return f"BlockDiagonalMatrix(n={self.n}, d={self.d})"
    
    def __str__(self) -> str:
        return f"BlockDiagonalMatrix(n={self.n}, d={self.d})\n{self.to_full_matrix()}"
    
    # ========================================
    # OPERATION 1: MATRIX MULTIPLICATION
    # ========================================
    
    def __matmul__(self, other: 'BlockDiagonalMatrix') -> 'BlockDiagonalMatrix':
        """
        Matrix multiplication: C = A @ B
        
        Key Insight:
        ------------
        For block matrices: C_ij = Σ_k A_ik @ B_kj
        
        For diagonal blocks: (D1 @ D2) is also diagonal!
        If D1 = diag(a) and D2 = diag(b), then D1 @ D2 = diag(a * b)
        
        This is the key property that makes our data structure work!
        
        Time Complexity: O(n³d) instead of O(n³d³) for naive approach
        
        Args:
            other: Another BlockDiagonalMatrix with same n, d
            
        Returns:
            Result of matrix multiplication
        """
        assert self.n == other.n and self.d == other.d, "Dimension mismatch"
        
        result_blocks = np.zeros((self.n, self.n, self.d))
        
        for i in range(self.n):
            for j in range(self.n):
                # C_ij = Σ_k A_ik @ B_kj
                # Since blocks are diagonal: diag(C_ij) = Σ_k diag(A_ik) * diag(B_kj)
                for k in range(self.n):
                    # Element-wise multiplication of diagonal elements
                    result_blocks[i, j, :] += self.blocks[i, k, :] * other.blocks[k, j, :]
        
        return BlockDiagonalMatrix(self.n, self.d, result_blocks)
    
    def multiply(self, other: 'BlockDiagonalMatrix') -> 'BlockDiagonalMatrix':
        """Alias for matrix multiplication."""
        return self @ other
    
    # ========================================
    # OPERATION 2: MATRIX INVERSE
    # ========================================
    
    def inverse(self) -> 'BlockDiagonalMatrix':
        """
        Compute the matrix inverse: A^{-1}
        
        Algorithm: Block LU Decomposition
        ----------------------------------
        We use block Gaussian elimination adapted for our structure.
        
        Since the blocks are diagonal, all block operations preserve
        the diagonal structure:
        - D1 + D2 is diagonal
        - D1 * D2 is diagonal (element-wise multiplication of diagonals)
        - D^{-1} is diagonal (element-wise reciprocal of diagonals)
        
        We compute A^{-1} by solving AX = I using block forward/backward substitution.
        
        Time Complexity: O(n³d) instead of O(n³d³)
        
        Returns:
            Inverse of the matrix
            
        Raises:
            np.linalg.LinAlgError: If matrix is singular
        """
        # Convert to a special representation for block Gaussian elimination
        # We'll use the standard approach: augment [A | I] and reduce to [I | A^{-1}]
        
        n, d = self.n, self.d
        
        # Create augmented matrix blocks: [A | I]
        # Left half is A, right half is I
        aug_left = self.blocks.copy()
        aug_right = np.zeros((n, n, d))
        for i in range(n):
            aug_right[i, i, :] = 1.0  # Identity matrix
        
        # Block Gaussian Elimination with partial pivoting
        for pivot_row in range(n):
            # Find best pivot (largest diagonal norm in column)
            best_row = pivot_row
            best_norm = np.linalg.norm(aug_left[pivot_row, pivot_row, :])
            
            for row in range(pivot_row + 1, n):
                norm = np.linalg.norm(aug_left[row, pivot_row, :])
                if norm > best_norm:
                    best_norm = norm
                    best_row = row
            
            # Swap rows if necessary
            if best_row != pivot_row:
                aug_left[[pivot_row, best_row], :, :] = aug_left[[best_row, pivot_row], :, :]
                aug_right[[pivot_row, best_row], :, :] = aug_right[[best_row, pivot_row], :, :]
            
            # Check for singularity
            pivot_diag = aug_left[pivot_row, pivot_row, :]
            if np.any(np.abs(pivot_diag) < 1e-10):
                raise np.linalg.LinAlgError("Matrix is singular or near-singular")
            
            # Normalize pivot row: divide by pivot block
            pivot_inv = 1.0 / pivot_diag  # Inverse of diagonal block = reciprocal of elements
            
            for col in range(n):
                aug_left[pivot_row, col, :] *= pivot_inv
                aug_right[pivot_row, col, :] *= pivot_inv
            
            # Eliminate column in all other rows
            for row in range(n):
                if row != pivot_row:
                    factor = aug_left[row, pivot_row, :].copy()
                    for col in range(n):
                        aug_left[row, col, :] -= factor * aug_left[pivot_row, col, :]
                        aug_right[row, col, :] -= factor * aug_right[pivot_row, col, :]
        
        return BlockDiagonalMatrix(n, d, aug_right)
    
    def is_close(self, other: 'BlockDiagonalMatrix', rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if two BlockDiagonalMatrices are approximately equal."""
        return np.allclose(self.blocks, other.blocks, rtol=rtol, atol=atol)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the full matrix."""
        return (self.n * self.d, self.n * self.d)
    
    def storage_size(self) -> int:
        """Return number of elements stored (for efficiency comparison)."""
        return self.n * self.n * self.d
    
    def full_matrix_size(self) -> int:
        """Return number of elements in full matrix representation."""
        return (self.n * self.d) ** 2
    
    def storage_efficiency(self) -> float:
        """Return storage efficiency ratio (lower is better)."""
        return self.storage_size() / self.full_matrix_size()


# ============================================
# VERIFICATION AND TESTING
# ============================================

def test_multiplication():
    """Test matrix multiplication correctness."""
    print("=" * 60)
    print("TEST 1: Matrix Multiplication")
    print("=" * 60)
    
    n, d = 4, 3
    
    # Create two random block-diagonal matrices
    A = BlockDiagonalMatrix.random(n, d, seed=42)
    B = BlockDiagonalMatrix.random(n, d, seed=123)
    
    # Compute using our efficient method
    C_efficient = A @ B
    
    # Compute using full matrices (for verification)
    A_full = A.to_full_matrix()
    B_full = B.to_full_matrix()
    C_full_expected = A_full @ B_full
    
    # Convert result back for comparison
    C_full_computed = C_efficient.to_full_matrix()
    
    # Check correctness
    max_error = np.max(np.abs(C_full_computed - C_full_expected))
    is_correct = max_error < 1e-10
    
    print(f"Matrix dimensions: {n*d} × {n*d}")
    print(f"Block structure: {n} × {n} blocks of size {d} × {d}")
    print(f"Maximum error: {max_error:.2e}")
    print(f"Test PASSED: {is_correct}")
    print()
    
    return is_correct


def test_inverse():
    """Test matrix inverse correctness."""
    print("=" * 60)
    print("TEST 2: Matrix Inverse")
    print("=" * 60)
    
    n, d = 3, 4
    
    # Create a random matrix (ensure it's invertible by adding to identity)
    A = BlockDiagonalMatrix.random(n, d, seed=42)
    # Make it diagonally dominant to ensure invertibility
    for i in range(n):
        A.blocks[i, i, :] += 10.0  # Add large value to diagonal blocks
    
    # Compute inverse
    A_inv = A.inverse()
    
    # Verify: A @ A^{-1} should equal I
    I_computed = A @ A_inv
    I_expected = BlockDiagonalMatrix.identity(n, d)
    
    max_error = np.max(np.abs(I_computed.to_full_matrix() - I_expected.to_full_matrix()))
    is_correct = max_error < 1e-8
    
    print(f"Matrix dimensions: {n*d} × {n*d}")
    print(f"Block structure: {n} × {n} blocks of size {d} × {d}")
    print(f"Maximum error in A @ A^(-1) - I: {max_error:.2e}")
    print(f"Test PASSED: {is_correct}")
    print()
    
    # Also verify using numpy's inverse on full matrix
    A_full = A.to_full_matrix()
    A_inv_full = np.linalg.inv(A_full)
    A_inv_computed_full = A_inv.to_full_matrix()
    
    max_error_vs_numpy = np.max(np.abs(A_inv_computed_full - A_inv_full))
    print(f"Max error vs NumPy's inverse: {max_error_vs_numpy:.2e}")
    print()
    
    return is_correct


def test_efficiency():
    """Demonstrate storage and computational efficiency."""
    print("=" * 60)
    print("TEST 3: Efficiency Analysis")
    print("=" * 60)
    
    configs = [(10, 5), (20, 10), (50, 20)]
    
    print(f"{'n':>5} {'d':>5} {'Full Size':>15} {'Our Size':>15} {'Efficiency':>12}")
    print("-" * 55)
    
    for n, d in configs:
        A = BlockDiagonalMatrix.random(n, d)
        full_size = A.full_matrix_size()
        our_size = A.storage_size()
        efficiency = A.storage_efficiency()
        
        print(f"{n:>5} {d:>5} {full_size:>15,} {our_size:>15,} {efficiency:>11.4f}x")
    
    print()
    print("Storage efficiency = (Our storage) / (Full matrix storage)")
    print("Lower is better. We achieve O(1/d) efficiency!")
    print()


def test_large_scale():
    """Test with larger matrices to verify scalability."""
    print("=" * 60)
    print("TEST 4: Large-Scale Test")
    print("=" * 60)
    
    import time
    
    n, d = 100, 50  # 5000 × 5000 matrix
    
    print(f"Testing with {n*d} × {n*d} matrix ({n}×{n} blocks of size {d}×{d})")
    print(f"Full matrix would require {(n*d)**2:,} elements")
    print(f"Our structure stores only {n*n*d:,} elements")
    print(f"Storage reduction: {(n*d)**2 / (n*n*d):.1f}x")
    print()
    
    A = BlockDiagonalMatrix.random(n, d, seed=42)
    B = BlockDiagonalMatrix.random(n, d, seed=123)
    
    # Time multiplication
    start = time.time()
    C = A @ B
    mult_time = time.time() - start
    print(f"Multiplication time: {mult_time:.4f} seconds")
    
    # For inverse, use smaller matrix
    n_inv, d_inv = 20, 10
    A_small = BlockDiagonalMatrix.random(n_inv, d_inv, seed=42)
    for i in range(n_inv):
        A_small.blocks[i, i, :] += 5.0  # Ensure invertibility
    
    start = time.time()
    A_inv = A_small.inverse()
    inv_time = time.time() - start
    print(f"Inverse time ({n_inv*d_inv}×{n_inv*d_inv}): {inv_time:.4f} seconds")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BLOCK-DIAGONAL MATRIX DATA STRUCTURE")
    print("Technical Challenge Solution")
    print("=" * 60 + "\n")
    
    # Run all tests
    test1_passed = test_multiplication()
    test2_passed = test_inverse()
    test_efficiency()
    test_large_scale()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Multiplication test: {'PASSED ✓' if test1_passed else 'FAILED ✗'}")
    print(f"Inverse test: {'PASSED ✓' if test2_passed else 'FAILED ✗'}")
    print()
    print("Key Achievements:")
    print("• Storage: O(n²d) instead of O(n²d²) — d times more efficient")
    print("• Multiplication: O(n³d) instead of O(n³d³) — d² times faster")
    print("• Inverse: O(n³d) using block Gaussian elimination")
    print()
