import functools

class PartitionManager(object):
    """
    Requires the input *instance* to have:
        1. self.exec_labels = set() or []
            store all the labels that needs to be executed
        2. self.args = {}  
            store all the intermediate tensors

    USAGE:
        1. skim through the model by setting filtering=False
        2. executes selected layers by setting filtering=True

    Specify the input args of the function as *the String keys* in instance.args;
    (when skimming)
        The function AUTOMATICALLY generates outputs into instance.args;
    (when filtering)
        The function only executes layers that are in instance.exec_labels

    Need to MANUALLY specify *suffix* for the namings of:
        1. intermediate tensors that are loaded from instance.args
        2. intermediate tensors that are stored into instance.args
    """
    def __init__(self, func, instance, filtering, suffix):
        self.func = func
        self.instance  = instance
        self.filtering = filtering
        self.label = self.func.__name__ + suffix

        functools.update_wrapper(self, func)

    def __call__(self, *args):
        if self.filtering:
            if self.label in self.instance.exec_labels:
                print("in exec", self.label)
                args = [self.instance.args[arg] for arg in args]
                return self.func(*args)
            else:
                print("not in exec")
        else:
            print("skimming", self.label)
            args = [self.instance.args[arg] for arg in args]
            self.instance.args[self.label] = self.func(*args)
            
            # # FIXME: MODIFY THIS SEGMENT
            # # Get all layer names
            # self.instance.exec_labels.add(self.label)

def partition_manager(instance, filtering, suffix):
    def partition_manager_decorator(func):
        return PartitionManager(func, instance, filtering, suffix)
    return partition_manager_decorator
