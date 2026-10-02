from types import SimpleNamespace,ModuleType
import sys
import pytest
from maniguard.eval.recording import make_observer

def test_disabled_observer_imports_nothing():
    assert make_observer(SimpleNamespace(recording_factory=None,recording_output_dir=None),{'name':'x'}) is None

def test_observer_factory_is_explicit_and_receives_unmodified_configuration(monkeypatch):
    module=ModuleType('test_observer_plugin');seen={};observer=object()
    def factory(**kw):seen.update(kw);return observer
    module.factory=factory;monkeypatch.setitem(sys.modules,module.__name__,module)
    cfg=SimpleNamespace(recording_factory='test_observer_plugin:factory',recording_output_dir='/tmp/recording')
    scene={'name':'task_0000/base'}
    assert make_observer(cfg,scene) is observer
    assert seen=={'cfg':cfg,'scene':scene}

def test_output_directory_without_factory_is_an_error():
    with pytest.raises(ValueError):make_observer(SimpleNamespace(recording_factory=None,recording_output_dir='x'),{})

def test_actual_execution_loop_preserves_actions_and_records_in_order():
    import ast,copy,io
    from pathlib import Path
    import numpy as np
    from contextlib import redirect_stdout
    tree=ast.parse((Path(__file__).resolve().parents[2]/'maniguard/eval/benchmark.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.While) and 'step_idx < cfg.max_steps' in ast.unparse(n.test))
    code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'actual-eval-loop','exec')
    outputs=[]
    for enabled in (False,True):
        executed=[];events=[]
        class Observer:
            def proposal(self,t,chunk,n,bounds):events.append(('proposal',t,n))
            def before_action(self,t,offset,raw,transformed,command):events.append(('before',t))
            def applied(self,t):events.append(('applied',t))
            def transition(self,t,obs):events.append(('state',t))
            def oracle(self,t,monitor):events.append(('oracle',t))
        def step(action):executed.append(action.copy());return None,0.,None,None,None
        def obs():return {'states':np.zeros(8),'overview_image':None,'wrist_images':None}
        chunk=np.ones((8,8),dtype=np.float32)*.2;chunk[:,-1]=.3
        scope={'cfg':SimpleNamespace(max_steps=4,execute_horizon=2,gripper_binarize=True,ik_eef_to_joint=False,save_video=False),
            'step_idx':0,'done':False,'success':False,'query_policy':lambda *args:chunk.copy(),'policy':None,'client_type':'fixture',
            'obs':obs(),'os':SimpleNamespace(environ={}), 'np':np,'env':SimpleNamespace(step=step),
            'robot':None,'scene_info':{'prompt':'move'},'episode_seed':7,'total_reward':0.,'recorder':Observer() if enabled else None,
            'action_space':SimpleNamespace(low=np.full(8,-1),high=np.ones(8),shape=(8,)),
            'torch':SimpleNamespace(from_numpy=lambda a:SimpleNamespace(unsqueeze=lambda dim:a)),
            'extract_obs':lambda *a:obs(),'rollout_diagnostics':{'action_attempts':0,'env_steps_returned':0,'observations_returned':0},
            '_record_action_finiteness':lambda *a:None,'ever_contacted':False,'_task_objs':[],'_target_obj':None,
            'monitor':SimpleNamespace(step=lambda t:None),'goal_checker':None,'goal_detail':{}}
        with redirect_stdout(io.StringIO()):exec(code,scope)
        outputs.append(np.stack(executed))
        if enabled:
            assert events==[('proposal',0,2),('before',0),('applied',0),('state',1),('oracle',1),('before',1),('applied',1),('state',2),('oracle',2),
                ('proposal',2,2),('before',2),('applied',2),('state',3),('oracle',3),('before',3),('applied',3),('state',4),('oracle',4)]
    np.testing.assert_array_equal(outputs[0],outputs[1]);assert outputs[0].shape==(4,8)
    np.testing.assert_array_equal(outputs[0][:,-1],np.ones(4))
