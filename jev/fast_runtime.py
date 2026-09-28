"""训练加速后端切换与真实实现诊断，默认保留 Transformers 原生分派。"""
import inspect
import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
_NATIVE_CHUNK=qmod.torch_chunk_gated_delta_rule
_NATIVE_CONV=qmod.causal_conv1d_fn
COUNTERS={'delta_calls':0,'conv_calls':0}


def implementation_info(function):
    seen=set()
    while callable(function) and id(function) not in seen:
        seen.add(id(function))
        try:
            closure=inspect.getclosurevars(function).nonlocals
        except (TypeError,ValueError):
            closure={}
        if callable(closure.get('implementation')):
            function=closure['implementation']
            break
        next_function=getattr(function,'__wrapped__',None)
        if next_function is None:
            break
        function=next_function
    return {'module':getattr(function,'__module__',None),'name':getattr(function,'__name__',None)}


def configure_delta_backend(mode='native'):
    COUNTERS.update(delta_calls=0,conv_calls=0)
    if mode=='native':
        qmod.torch_chunk_gated_delta_rule=_NATIVE_CHUNK
        qmod.causal_conv1d_fn=_NATIVE_CONV
    elif mode=='fla':
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        def chunk(query,key,value,g,beta,chunk_size=64,initial_state=None,
                  output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
            COUNTERS['delta_calls']+=1
            return chunk_gated_delta_rule(q=query,k=key,v=value,g=g,beta=beta,
                initial_state=initial_state,output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=kwargs.get('cu_seqlens'))
        qmod.torch_chunk_gated_delta_rule=chunk
        try:
            from causal_conv1d import causal_conv1d_fn
        except ImportError:
            qmod.causal_conv1d_fn=_NATIVE_CONV
        else:
            def conv(x,weight,bias=None,activation=None,**kwargs):
                COUNTERS['conv_calls']+=1
                return causal_conv1d_fn(x.to(weight.dtype),weight,bias,activation=activation).to(x.dtype)
            qmod.causal_conv1d_fn=conv
    else:
        raise ValueError('JEV_DELTA_BACKEND 仅支持 native 或 fla')
    return {'mode':mode,'chunk':implementation_info(qmod.torch_chunk_gated_delta_rule),
            'conv':implementation_info(qmod.causal_conv1d_fn)}
