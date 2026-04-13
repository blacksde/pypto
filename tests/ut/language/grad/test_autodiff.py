# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Unit tests for automatic differentiation functionality.
"""

import pytest

import pypto.language as pl
from pypto.ir.grad_registry import GradRegistry
from pypto.language.grad.autodiff_engine import AutodiffEngine, get_global_autodiff_engine


class TestGradRegistry:
    """Test gradient registry functionality."""
    
    def test_singleton_pattern(self):
        """Test that GradRegistry follows singleton pattern."""
        registry1 = GradRegistry.get_instance()
        registry2 = GradRegistry.get_instance()
        assert registry1 is registry2
    
    def test_register_grad(self):
        """Test registering a gradient function."""
        registry = GradRegistry.get_instance()
        
        def dummy_grad(inputs, grad_output):
            return [grad_output]
        
        registry.register_grad("test.op", dummy_grad)
        assert registry.has_grad("test.op")
    
    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate gradient raises error."""
        registry = GradRegistry.get_instance()
        
        def dummy_grad(inputs, grad_output):
            return [grad_output]
        
        registry.register_grad("test.duplicate", dummy_grad)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register_grad("test.duplicate", dummy_grad)
    
    def test_get_grad(self):
        """Test retrieving a gradient function."""
        registry = GradRegistry.get_instance()
        
        def dummy_grad(inputs, grad_output):
            return [grad_output]
        
        registry.register_grad("test.get", dummy_grad)
        retrieved = registry.get_grad("test.get")
        assert retrieved is not None
        assert retrieved is dummy_grad
    
    def test_get_nonexistent_grad(self):
        """Test getting non-existent gradient returns None."""
        registry = GradRegistry.get_instance()
        retrieved = registry.get_grad("nonexistent.op")
        assert retrieved is None
    
    def test_list_registered_ops(self):
        """Test listing all registered operators."""
        registry = GradRegistry.get_instance()
        
        def dummy_grad(inputs, grad_output):
            return [grad_output]
        
        registry.register_grad("test.list1", dummy_grad)
        registry.register_grad("test.list2", dummy_grad)
        
        ops = registry.list_registered_ops()
        assert "test.list1" in ops
        assert "test.list2" in ops


class TestAutodiffEngine:
    """Test automatic differentiation engine."""
    
    def test_singleton_pattern(self):
        """Test that global autodiff engine follows singleton pattern."""
        engine1 = get_global_autodiff_engine()
        engine2 = get_global_autodiff_engine()
        assert engine1 is engine2
    
    def test_start_stop_recording(self):
        """Test starting and stopping recording."""
        engine = AutodiffEngine()
        
        assert not engine.is_recording
        engine.start_recording()
        assert engine.is_recording
        engine.stop_recording()
        assert not engine.is_recording
    
    def test_record_operation(self):
        """Test recording operations in computation graph."""
        engine = AutodiffEngine()
        engine.start_recording()
        
        # Create dummy expressions
        from pypto.pypto_core.ir import Var
        x = Var("x")
        y = Var("y")
        z = Var("z")
        
        engine.record_operation("test.add", [x, y], z)
        
        assert len(engine.computation_graph) == 1
        assert engine.computation_graph[0]['op_name'] == "test.add"
        assert engine.computation_graph[0]['inputs'] == [x, y]
        assert engine.computation_graph[0]['output'] == z
    
    def test_record_operation_when_not_recording(self):
        """Test that operations are not recorded when not recording."""
        engine = AutodiffEngine()
        
        from pypto.pypto_core.ir import Var
        x = Var("x")
        y = Var("y")
        z = Var("z")
        
        engine.record_operation("test.add", [x, y], z)
        
        assert len(engine.computation_graph) == 0
    
    def test_clear(self):
        """Test clearing computation graph and gradient map."""
        engine = AutodiffEngine()
        engine.start_recording()
        
        from pypto.pypto_core.ir import Var
        x = Var("x")
        y = Var("y")
        z = Var("z")
        
        engine.record_operation("test.add", [x, y], z)
        engine.gradient_map[x] = y
        
        assert len(engine.computation_graph) == 1
        assert len(engine.gradient_map) == 1
        
        engine.clear()
        
        assert len(engine.computation_graph) == 0
        assert len(engine.gradient_map) == 0


class TestRegisterGradDecorator:
    """Test @pl.register_grad decorator."""
    
    def test_register_grad_decorator(self):
        """Test that decorator registers gradient function."""
        @pl.register_grad("test.decorator")
        def custom_grad(inputs, grad_output):
            return [grad_output]
        
        registry = GradRegistry.get_instance()
        assert registry.has_grad("test.decorator")
        
        retrieved = registry.get_grad("test.decorator")
        assert retrieved is custom_grad


class TestAutoRegistration:
    """Test automatic gradient registration."""
    
    def test_tensor_ops_have_gradients(self):
        """Test that common tensor ops have registered gradients."""
        registry = GradRegistry.get_instance()
        
        # Check that common tensor operations have gradients
        tensor_ops_with_grads = [
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
        
        for op_name in tensor_ops_with_grads:
            assert registry.has_grad(op_name), f"Missing gradient for {op_name}"
    
    def test_tile_ops_have_gradients(self):
        """Test that common tile ops have registered gradients."""
        registry = GradRegistry.get_instance()
        
        # Check that common tile operations have gradients
        tile_ops_with_grads = [
            "tile.add",
            "tile.sub",
            "tile.mul",
            "tile.div",
            "tile.neg",
        ]
        
        for op_name in tile_ops_with_grads:
            assert registry.has_grad(op_name), f"Missing gradient for {op_name}"


class TestGradientComputation:
    """Test gradient computation for simple functions."""
    
    def test_add_gradient(self):
        """Test gradient computation for add operation."""
        registry = GradRegistry.get_instance()
        
        # Register test gradient
        @pl.register_grad("test.simple_add")
        def grad_add(inputs, grad_output):
            return [grad_output, grad_output]
        
        # Create engine and record operation
        engine = AutodiffEngine()
        engine.start_recording()
        
        from pypto.pypto_core.ir import Var
        x = Var("x")
        y = Var("y")
        z = Var("z")
        
        engine.record_operation("test.simple_add", [x, y], z)
        engine.stop_recording()
        
        # Compute gradients
        grad_z = Var("grad_z")
        gradients = engine.compute_gradients([z], [grad_z])
        
        assert gradients[x] == grad_z
        assert gradients[y] == grad_z
    
    def test_mul_gradient(self):
        """Test gradient computation for mul operation."""
        registry = GradRegistry.get_instance()
        
        # Register test gradient
        @pl.register_grad("test.simple_mul")
        def grad_mul(inputs, grad_output):
            x, y = inputs
            return [pl.mul(grad_output, y), pl.mul(x, grad_output)]
        
        # Create engine and record operation
        engine = AutodiffEngine()
        engine.start_recording()
        
        from pypto.pypto_core.ir import Var
        x = Var("x")
        y = Var("y")
        z = Var("z")
        
        engine.record_operation("test.simple_mul", [x, y], z)
        engine.stop_recording()
        
        # Compute gradients
        grad_z = Var("grad_z")
        gradients = engine.compute_gradients([z], [grad_z])
        
        # Verify gradients are computed
        assert x in gradients
        assert y in gradients


class TestGradContextManager:
    """Test gradient context management."""
    
    def test_enable_disable_grad(self):
        """Test enabling and disabling gradient recording."""
        assert not pl.is_grad_enabled()
        
        pl.enable_grad()
        assert pl.is_grad_enabled()
        
        pl.disable_grad()
        assert not pl.is_grad_enabled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
