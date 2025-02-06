try:
    from hbdk4.compiler import Module as HbirModule
except ImportError:

    class HbirModule:
        pass
