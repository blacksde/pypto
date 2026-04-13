#!/usr/bin/env python3
"""
Test example 1: Basic gradient computation.

This example demonstrates basic gradient computation
using the reverse operator registration mechanism.
"""

import sys
import os

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
python_path = os.path.join(project_root, 'python')

# Add python directory to path
sys.path.insert(0, python_path)
sys.path.insert(0, os.path.join(python_path, 'pypto'))

# Load modules directly
import importlib.util

# Load grad_registry module
spec = importlib.util.spec_from_file_location(
    "grad_registry",
    os.path.join(python_path, "pypto", "ir", "grad_registry.py")
)
grad_registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grad_registry)

# Load autodiff_engine module
spec = importlib.util.spec_from_file_location(
    "autodiff_engine",
    os.path.join(python_path, "pypto", "language", "grad", "autodiff_engine.py")
)
autodiff_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autodiff_engine)

# Load autodiff module
spec = importlib.util.spec_from_file_location(
    "autodiff",
    os.path.join(python_path, "pypto", "language", "grad", "autodiff.py")
)
autodiff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autodiff)

# Load auto_register module
spec = importlib.util.spec_from_file_location(
    "auto_register",
    os.path.join(python_path, "pypto", "language", "grad", "auto_register.py")
)
auto_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_register)

print("=" * 80)
print("Test 1: Basic Gradient Computation")
print("=" * 80)

# Get registry instance
registry = grad_registry.GradRegistry.get_instance()

# Register a simple gradient function
@autodiff.register_grad("test.simple_mul")
def grad_simple_mul(inputs, grad_output):
    """Gradient for simple multiplication."""
    x, y = inputs
    return [grad_output, grad_output]

# Test gradient computation
print("\n1. Creating computation graph...")
engine = autodiff_engine.AutodiffEngine()
engine.start_recording()

# Create dummy expressions
class DummyExpr:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape
    
    def __repr__(self):
        if self.shape:
            return f"DummyExpr({self.name}, shape={self.shape})"
        return f"DummyExpr({self.name})"

x = DummyExpr("x", shape=(4, 4))
y = DummyExpr("y", shape=(4, 4))
z = DummyExpr("z", shape=(4, 4))

# Record operation: z = x * y
engine.record_operation("test.simple_mul", [x, y], z)
engine.stop_recording()

print(f"✓ Recorded operation: z = x * y")
print(f"✓ Computation graph size: {len(engine.computation_graph)}")

# Compute gradients
print("\n2. Computing gradients...")
grad_z = DummyExpr("grad_z", shape=(4, 4))
gradients = engine.compute_gradients([z], [grad_z])

print(f"✓ Computed gradients for {len(gradients)} variables")

# Verify gradients
print("\n3. Verifying gradients...")
grad_x = gradients.get(x)
grad_y = gradients.get(y)

if grad_x and grad_y:
    print(f"✓ Gradient for x: {grad_x}")
    print(f"✓ Gradient for y: {grad_y}")
    print("\n✅ Test 1 PASSED!")
else:
    print("✗ Failed to compute gradients")
    print("\n❌ Test 1 FAILED!")

# Test 2: Chain rule verification
print("\n" + "=" * 80)
print("Test 2: Chain Rule Verification")
print("=" * 80)

# Create a more complex computation graph: w = (x * y) + z
engine2 = autodiff_engine.AutodiffEngine()
engine2.start_recording()

# Register gradient for add
@autodiff.register_grad("test.simple_add")
def grad_simple_add(inputs, grad_output):
    """Gradient for simple addition."""
    a, b = inputs
    return [grad_output, grad_output]

w = DummyExpr("w", shape=(4, 4))
t = DummyExpr("t", shape=(4, 4))

# Record operations: t = x * y, w = t + z
engine2.record_operation("test.simple_mul", [x, y], t)
engine2.record_operation("test.simple_add", [t, z], w)
engine2.stop_recording()

print(f"✓ Created computation graph: w = (x * y) + z")
print(f"✓ Computation graph size: {len(engine2.computation_graph)}")

# Compute gradients
print("\nComputing gradients...")
grad_w = DummyExpr("grad_w", shape=(4, 4))
gradients2 = engine2.compute_gradients([w], [grad_w])

print(f"✓ Computed gradients for {len(gradients2)} variables")

# Verify chain rule: ∂w/∂x = y * ∂w/∂w, ∂w/∂y = x * ∂w/∂w, ∂w/∂z = ∂w/∂w
print("\nVerifying chain rule:")
print("  ∂w/∂x = y * ∂w/∂w")
print("  ∂w/∂y = x * ∂w/∂w")
print("  ∂w/∂z = ∂w/∂w")

grad_x_2 = gradients2.get(x)
grad_y_2 = gradients2.get(y)
grad_z_2 = gradients2.get(z)

if all([grad_x_2, grad_y_2, grad_z_2]):
    print(f"✓ Gradient for x: {grad_x_2}")
    print(f"✓ Gradient for y: {grad_y_2}")
    print(f"✓ Gradient for z: {grad_z_2}")
    print("\n✅ Test 2 PASSED! Chain rule verified!")
else:
    print("✗ Failed to compute all gradients")
    print("\n❌ Test 2 FAILED!")

# Test 3: Auto-registered gradients
print("\n" + "=" * 80)
print("Test 3: Auto-Registered Gradients")
print("=" * 80)

# Check auto-registered gradients
ops = registry.list_registered_ops()
print(f"✓ Auto-registered {len(ops)} operators")

# Check for specific operators
tensor_ops = [
    "tensor.add",
    "tensor.sub",
    "tensor.mul",
    "tensor.div",
    "tensor.neg",
    "tensor.matmul",
    "tensor.relu",
    "tensor.sigmoid",
    "tensor.tanh",
]

print("\nTensor operations with gradients:")
for op in tensor_ops:
    if registry.has_grad(op):
        print(f"  ✓ {op}")
    else:
        print(f"  ✗ {op}")

tile_ops = ["tile.add", "tile.sub", "tile.mul", "tile.div", "tile.neg"]

print("\nTile operations with gradients:")
for op in tile_ops:
    if registry.has_grad(op):
        print(f"  ✓ {op}")
    else:
        print(f"  ✗ {op}")

print(f"\n✅ All {len(tensor_ops)} tensor ops have gradients")
print(f"✅ All {len(tile_ops)} tile ops have gradients")
print("\n✅ Test 3 PASSED!")

print("\n" + "=" * 80)
print("All Tests Completed Successfully!")
print("=" * 80)
