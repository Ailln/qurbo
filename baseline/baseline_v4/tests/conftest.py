import pytest
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from baseline.baseline_v4.data.instance import MIQPInstance
from baseline.baseline_v4.core.evaluator import ObjectiveEvaluator

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'alpha-test')


@pytest.fixture
def instance_a():
    path = os.path.join(DATA_DIR, 'miqp_sample_A.npz')
    inst = MIQPInstance()
    inst.load(path)
    inst.validate()
    return inst


@pytest.fixture
def instance_b():
    path = os.path.join(DATA_DIR, 'miqp_sample_B.npz')
    inst = MIQPInstance()
    inst.load(path)
    inst.validate()
    return inst


@pytest.fixture
def evaluator_a(instance_a):
    return ObjectiveEvaluator(instance_a)
